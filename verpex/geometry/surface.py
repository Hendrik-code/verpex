import nibabel as nib
import numpy as np
import torch
from skimage import measure

# from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F  # noqa: N812

# from BIDS import NII, POI
from TPTBox import NII
from TPTBox.core.np_utils import np_fill_holes
from TPTBox.core.poi import POI

from verpex.paths import get_path

# from utils.raycast_torch import max_distance_ray_cast_convex_torch


def _debug_path(name: str) -> str:
    """Return a path under the configured scratch directory for a debug dump.

    Only called when a `debug` flag is set; creates the directory on first use.
    """
    directory = get_path("tmp_root") / "debug"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / name)


def np_to_bids_nii(array: np.ndarray) -> NII:
    """Converts a numpy array to a BIDS NII object."""
    # NiBabel expects the orientation to be RAS+ (right, anterior, superior, plus),
    # we have LAS+ (left, posterior, superior, plus) so we need to flip along the second axis
    affine = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    nifty1img = nib.Nifti1Image(array, affine)
    return NII(nifty1img)


def surface_project_poi_vert_wise(poi: POI, surface_nii: NII, requires_filling: bool = True) -> POI:
    """Project a multi-vertebra POI onto the surface, one vertebra at a time.

    Projecting per vertebra keeps each landmark on its own bone: a whole-spine
    projection would let a landmark snap to a neighbouring vertebra's surface
    wherever two vertebrae nearly touch.

    Args:
        poi: Landmarks spanning one or more vertebrae.
        surface_nii: Surface mask labelled by vertebra.
        requires_filling: Fill interior holes in the mask before projecting.

    Returns:
        A new POI with every landmark projected onto its own vertebra's surface.

    Raises:
        AssertionError: If a vertebra present in ``poi`` is absent from ``surface_nii``.
    """
    poi_new = poi.make_empty_POI()
    for v in poi.keys_region():
        assert v in surface_nii.unique(), f"Surface NII does not contain vertebra {v}"

        # extract surface for this vertebra
        surf_v_nii = surface_nii.extract_label(v)
        poi_v = poi.extract_region(v)
        poi_v_proj = surface_project_poi(poi_v, surf_v_nii, requires_filling=requires_filling)
        for r, s, c in poi_v_proj.items():
            poi_new[r, s] = c
    return poi_new


def surface_project_poi(
    poi: POI,
    surface_nii: NII,
    requires_filling: bool = True,
    debug: bool = False,
) -> POI:
    """Project a single vertebra's landmarks onto its surface mask.

    Works on a crop around the surface for speed, then maps the result back into
    the original frame.

    Args:
        poi: Landmarks for one vertebra.
        surface_nii: Binary surface mask for that vertebra.
        requires_filling: Fill interior holes in the mask before projecting.
        debug: Write intermediate masks to the configured scratch directory.

    Returns:
        A new POI with projected coordinates, in the input's frame.
    """
    crop = surface_nii.compute_crop(dist=4)
    poi_crop = poi.apply_crop(crop)
    # convert poi to coordinates tensor
    coord_keys = []
    coords_list = []
    for r, s, c in poi_crop.items():
        coord_keys.append((r, s))
        coords_list.append(c)
    # surface_nii to torch tensor
    surface_tensor = torch.from_numpy(surface_nii.apply_crop(crop).get_array()).unsqueeze(0).to(torch.float32)  # (1,D,H,W)
    # run surface_project_coords
    coords_tensor = torch.from_numpy(np.asarray(coords_list)).unsqueeze(0).to(torch.float32)  # (1,N,3)
    projected_coords, _ = surface_project_coords(
        coords_tensor,
        surface_tensor,
        requires_filling=requires_filling,
        debug=debug,
    )  # (1,N,3)
    # transfer back
    projected_poi = poi_crop.make_empty_POI()
    for i, (r, s) in enumerate(coord_keys):
        projected_poi[r, s] = projected_coords[0, i].cpu().numpy()
    final_poi = projected_poi.apply_crop_reverse(crop, surface_nii.shape)
    # return
    return final_poi


def surface_project_coords(
    coordinates,
    surface,
    debug=False,
    requires_filling: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project predicted coordinates onto the surface of a binary mask.

    This is the projection used on the live training and inference path. It
    delegates to the continuous marching-cubes implementation, which returns
    sub-voxel positions on the reconstructed mesh rather than snapping to voxel
    centres.

    Args:
        coordinates: Coordinates to project, ``(batch, n_landmarks, 3)``.
        surface: Binary surface mask, ``(batch, depth, height, width)``.
        debug: Write intermediate masks to the configured scratch directory.
        requires_filling: Fill interior holes in the mask before projecting.

    Returns:
        A tuple of the projected coordinates and the distance each one moved.
    """
    return surface_project_coords_marchingcubes_continuous(coordinates, surface, debug=debug, requires_filling=requires_filling)


def fill_holes_3d_6conn(surface_mask) -> torch.Tensor:
    """Fill holes using strict 6-connectivity (no diagonal connectivity).

    Runs on GPU and does not overfill thin structures, unlike a 26-connected fill.

    surface_mask: (B,1,D,H,W) boolean or int tensor
        True = object/surface voxels

    Returns:
        filled: (B,1,D,H,W) boolean tensor
            original object + filled 6-connected cavities
    """
    surf = surface_mask.bool()
    _B, _, _D, _H, _W = surf.shape
    device = surf.device

    # Step 1 — solid voxels as-is
    solid = surf

    # Step 2 — outside = everything not solid
    outside = ~solid

    # Step 3 — Outside seeds = boundary outside voxels
    seeds = torch.zeros_like(outside)
    seeds[:, :, 0, :, :] = True
    seeds[:, :, -1, :, :] = True
    seeds[:, :, :, 0, :] = True
    seeds[:, :, :, -1, :] = True
    seeds[:, :, :, :, 0] = True
    seeds[:, :, :, :, -1] = True
    seeds = seeds & outside

    # --- 6-connected kernel (center + 6 face neighbors)
    kernel = torch.zeros((1, 1, 3, 3, 3), device=device)
    kernel[0, 0, 1, 1, 0] = 1  # -x
    kernel[0, 0, 1, 1, 2] = 1  # +x
    kernel[0, 0, 1, 0, 1] = 1  # -y
    kernel[0, 0, 1, 2, 1] = 1  # +y
    kernel[0, 0, 0, 1, 1] = 1  # -z
    kernel[0, 0, 2, 1, 1] = 1  # +z

    # Step 4 — flood-fill outside with 6-connectivity
    cur = seeds.clone()

    while True:
        # 6-connected dilation
        expanded = F.conv3d(cur.float(), kernel, padding=1) > 0

        # Only grow into true outside
        expanded = expanded & outside

        if torch.equal(expanded, cur):
            break

        cur = expanded

    outside_filled = cur
    inside = ~outside_filled  # object + closed cavities

    # Holes = inside but not originally solid
    holes = inside & (~solid)

    # Final = original + holes
    filled = solid | holes

    return filled


def extract_surface_vertices(mask, level=0.5) -> torch.Tensor:
    """Extract surface vertices from a binary mask via marching cubes.

    Args:
        mask: Binary mask, ``(depth, height, width)``.
        level: Iso-surface level passed to marching cubes.

    Returns:
        The surface vertex coordinates, ``(n_vertices, 3)``.
    """
    verts, _, _, _ = measure.marching_cubes(
        mask.astype(float),
        level=level,
    )
    return torch.from_numpy(verts.copy()).float()  # (M, 3)


def surface_project_coords_marchingcubes(  # noqa: ANN201
    coordinates,
    surface_mask,
    level=0.5,
    debug=False,
    requires_filling: bool = False,
):
    """Project coordinates onto the nearest marching-cubes surface vertex.

    Snaps to mesh vertices, so results land on discrete positions. Prefer
    :func:`surface_project_coords_marchingcubes_continuous` for sub-voxel accuracy.

    Args:
        coordinates: ``(batch, n_landmarks, 3)`` or ``(n_landmarks, 3)``.
        surface: Binary mask, ``(batch, depth, height, width)`` or ``(depth, height, width)``.

    Returns:
        A tuple of the projected coordinates and the distance each one moved.
    """
    # ---------- Handle batching ----------
    unbatched_coords = coordinates.ndim == 2
    unbatched_surface = surface_mask.ndim == 3

    if unbatched_coords:
        coordinates = coordinates.unsqueeze(0)
    if unbatched_surface:
        surface_mask = surface_mask.unsqueeze(0)

    B, _, _ = coordinates.shape
    device = coordinates.device

    # ---------- Extract marching cubes surfaces for each batch ----------
    surface_vertices = []
    for b in range(B):
        mask_np = surface_mask[b].detach().cpu().numpy().astype(float)
        if mask_np.ndim == 4:
            mask_np = mask_np[0]

        if debug:
            np_to_bids_nii(mask_np).save(_debug_path("mask_np.nii.gz"))
        if requires_filling:
            mask_np = np_fill_holes(mask_np)
        if debug:
            np_to_bids_nii(mask_np).save(_debug_path("mask_np_filled.nii.gz"))
        (verts,) = (extract_surface_vertices(mask_np, level=level),)
        surface_vertices.append(verts)

    # pad surfaces to same size so batching works
    max_M = max(v.shape[0] for v in surface_vertices)
    padded_surfaces = torch.zeros((B, max_M, 3), device=device, dtype=torch.float32)
    surface_valid = torch.zeros((B, max_M), device=device, dtype=torch.bool)

    for b, v in enumerate(surface_vertices):
        M = v.shape[0]
        padded_surfaces[b, :M] = v
        surface_valid[b, :M] = True

    # print shapes
    # print("padded surfaces shape:", padded_surfaces.shape)
    # print("coordinates shape:", coordinates.shape)
    # print("surface_valid shape:", surface_valid.shape)
    # print("surface_mask_shape:", surface_mask.shape)
    # print("coords example:", coordinates[0, :5])
    # print("surface example:", padded_surfaces[0, :5])

    # ---------- Compute distances ----------
    coords_exp = coordinates.unsqueeze(2)  # (B, N, 1, 3)
    surf_exp = padded_surfaces.unsqueeze(1)  # (B, 1, M, 3)
    diff = coords_exp - surf_exp  # (B, N, M, 3)
    dist_sq = (diff**2).sum(dim=-1)  # (B, N, M)

    # mask invalid padded values
    # dist_sq[~surface_valid.unsqueeze(1)] = float("inf")
    dist_sq[~surface_valid.unsqueeze(1).expand_as(dist_sq)] = float("inf")

    # ---------- Find nearest surface vertex ----------
    min_dist_sq, min_idx = torch.min(dist_sq, dim=-1)  # (B, N)

    # ---------- Gather vertices ----------
    batch_idx = torch.arange(B, device=device).unsqueeze(-1)
    projected = padded_surfaces[batch_idx, min_idx]  # (B, N, 3)

    # ---------- Convert to voxel indices ----------
    # surface_projected_targets = projected.round().long()
    surface_projection_dist = torch.sqrt(min_dist_sq)

    # ---------- Unbatch if needed ----------
    if unbatched_coords:
        projected = projected.squeeze(0)
        surface_projection_dist = surface_projection_dist.squeeze(0)

    return projected, surface_projection_dist


def closest_point_on_triangle_batch(p, a, b, c) -> torch.Tensor:
    """Batched closest point on triangles.

    p : (N,3)
    a,b,c : (K,3)

    Returns:
        proj : (N,K,3)
    """
    # expand for broadcasting
    p = p[:, None, :]  # (N,1,3)
    a = a[None, :, :]  # (1,K,3)
    b = b[None, :, :]
    c = c[None, :, :]

    ab = b - a  # (1,K,3)
    ac = c - a
    ap = p - a  # (N,K,3)

    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)

    # barycentric coordinates
    d00 = (ab * ab).sum(-1)
    d01 = (ab * ac).sum(-1)
    d11 = (ac * ac).sum(-1)

    denom = d00 * d11 - d01 * d01 + 1e-12

    v = (d11 * d1 - d01 * d2) / denom
    w = (d00 * d2 - d01 * d1) / denom
    u = 1.0 - v - w

    # clamp barycentric to triangle
    v = torch.clamp(v, 0.0, 1.0)
    w = torch.clamp(w, 0.0, 1.0)
    u = torch.clamp(u, 0.0, 1.0)

    norm = u + v + w + 1e-12
    u = u / norm
    v = v / norm
    w = w / norm

    proj = u[..., None] * a + v[..., None] * b + w[..., None] * c
    return proj


# -------------------------------------------------------------


def surface_project_coords_marchingcubes_continuous(  # noqa: ANN201
    coordinates,
    surface_mask,
    level=0.5,
    debug=False,  # noqa: ARG001 - kept so both projection variants share one signature
    requires_filling: bool = False,
):
    """Project coordinates onto the closest point of the marching-cubes mesh.

    Unlike the vertex-snapping variant, this finds the closest point on the mesh
    *triangles*, so the result is continuous and sub-voxel accurate. This is what
    :func:`surface_project_coords` uses.

    Args:
        coordinates: ``(batch, n_landmarks, 3)`` or ``(n_landmarks, 3)``.
        surface: Binary mask, ``(batch, depth, height, width)`` or ``(depth, height, width)``.

    Returns:
        A tuple of the projected coordinates and the distance each one moved.
    """
    # ---------------- batching ----------------
    unbatched_coords = coordinates.ndim == 2
    unbatched_surface = surface_mask.ndim == 3

    if unbatched_coords:
        coordinates = coordinates.unsqueeze(0)
    if unbatched_surface:
        surface_mask = surface_mask.unsqueeze(0)

    B, N, _ = coordinates.shape
    device = coordinates.device

    # --------- extract surfaces per batch (verts + faces) ----------
    surfaces = []
    max_M = 0
    max_K = 0

    for b in range(B):
        mask_np = surface_mask[b].detach().cpu().numpy()

        if mask_np.ndim == 4:
            mask_np = mask_np[0]

        if requires_filling:
            mask_np = np_fill_holes(mask_np)

        verts, faces, _, _ = measure.marching_cubes(
            mask_np.astype(np.uint8),
            level=level,
            allow_degenerate=False,
            step_size=1,
        )

        # verts = verts + 0.5  # shift from voxel corner to center

        verts = torch.from_numpy(verts.copy()).float().to(device)
        faces = torch.from_numpy(faces.copy()).long().to(device)
        # verts = verts[:, ::-1]  # convert to (x,y,z)

        surfaces.append((verts, faces))
        max_M = max(max_M, verts.shape[0])
        max_K = max(max_K, faces.shape[0])

    # -------- pad to batch tensors ----------
    verts_pad = torch.zeros((B, max_M, 3), device=device)
    faces_pad = torch.zeros((B, max_K, 3), device=device, dtype=torch.long)
    verts_valid = torch.zeros((B, max_M), device=device, dtype=torch.bool)
    faces_valid = torch.zeros((B, max_K), device=device, dtype=torch.bool)

    for b, (v, f) in enumerate(surfaces):
        M = v.shape[0]
        K = f.shape[0]

        verts_pad[b, :M] = v
        faces_pad[b, :K] = f
        verts_valid[b, :M] = True
        faces_valid[b, :K] = True

    # ------------- projection ----------------
    projected = torch.zeros_like(coordinates)
    surface_projection_dist = torch.zeros((B, N), device=device)

    for b in range(B):
        V = verts_pad[b]  # (M,3)
        F = faces_pad[b]
        Fmask = faces_valid[b]

        faces = F[Fmask]  # (K,3)

        v0 = V[faces[:, 0]]
        v1 = V[faces[:, 1]]
        v2 = V[faces[:, 2]]

        p = coordinates[b]  # (N,3)

        # ---- vectorized triangle projection ----
        proj_all = closest_point_on_triangle_batch(p, v0, v1, v2)  # (N,K,3)

        # distances
        diff = proj_all - p[:, None, :]
        dist2 = (diff**2).sum(-1)  # (N,K)

        best_k = dist2.argmin(dim=1)  # (N,)

        best_proj = proj_all[torch.arange(N), best_k]

        projected[b] = best_proj
        surface_projection_dist[b] = torch.sqrt(dist2[torch.arange(N), best_k])

    # -------- unbatch if needed ----------
    if unbatched_coords:
        projected = projected.squeeze(0)
        surface_projection_dist = surface_projection_dist.squeeze(0)

    return projected, surface_projection_dist


# POI Visualization
# Define some useful utility functions
