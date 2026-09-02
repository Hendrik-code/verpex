"""Loading and preparing the arrays a POI dataset serves.

BIDS file lookup, ground-truth POI extraction, heatmap rendering, surface computation
and the padding helpers that bring every cutout to a fixed shape.
"""

import os
from collections.abc import Callable
from os import PathLike

import numpy as np
import torch
from numpy import ndarray
from skimage.morphology import binary_erosion

# from BIDS import NII, POI
# from BIDS.bids_files import Subject_Container
from TPTBox import NII, Subject_Container
from TPTBox.core.poi import POI

#: Subregion labels of the vertebral body, used as the default one-hot channels.
VERTEBRA_BODY_SUBREGIONS = (41, 42, 43, 44, 45, 46, 47, 48, 49, 50)


def get_gt_pois(poi, vertebra, poi_indices):  # noqa: ANN201
    """Converts the POI coordinates to a tensor.

    Args:
        poi (POI): The POI coordinates.
        vertebra (int): The vertebra number.

    Returns:
        torch.Tensor: The POI coordinates as a tensor.
    """
    coords = [
        (np.array((-1, -1, -1)) if (vertebra, p_idx) not in poi.keys() else np.array(poi.centroids[vertebra, p_idx]))
        for p_idx in poi_indices
    ]

    # Stack the coordinates
    coords = np.stack(coords)

    # Change type of coords to float
    coords = coords.astype(np.float32)  # Shape: (n_pois, 3)

    # Mark the missing pois
    missing_poi_list_idx = np.all(coords == -1, axis=1)  # Shape: (n_pois,)

    # Get the indices of missing pois
    missing_pois = np.array([poi_idx for i, poi_idx in enumerate(poi_indices) if missing_poi_list_idx[i]])

    return torch.from_numpy(coords), torch.from_numpy(missing_pois)


def compute_surface(msk: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """Computes the surface of the vertebra.

    Args:
        msk: The segmentation mask.
        iterations: How many erosion steps to peel off before subtracting.

    Returns:
        torch.Tensor: The surface of the vertebra.
    """
    # `.numpy()` returns a view, so this must be copied before being written to;
    # otherwise the erosion below edits the caller's mask in place.
    surface = msk.numpy().copy()

    eroded = surface.copy()
    for _ in range(iterations):
        eroded = binary_erosion(eroded)

    surface[eroded] = 0

    return torch.from_numpy(surface)


def apply_dictionary_transform(
    transform: Callable, im: ndarray, subreg: ndarray, vertseg: ndarray, poi_hm: ndarray
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Apply a random affine transformation to the input and target.

    Args:
    - im: The input image.
    - subreg: The subregional segmentation.
    - vertseg: The vertebral segmentation.
    - poi_hm: The heatmap of the points of interest.
    """
    # Add channel dimension to the input
    im = np.expand_dims(im, axis=0)
    subreg = np.expand_dims(subreg, axis=0)
    vertseg = np.expand_dims(vertseg, axis=0)

    # Create a dictionary with the input and target
    data_dict = {"im": im, "subreg": subreg, "vertseg": vertseg, "target": poi_hm}

    transformed_data_dict = transform(data_dict)

    # Convert back to numpy
    im = transformed_data_dict["im"]
    subreg = transformed_data_dict["subreg"]
    vertseg = transformed_data_dict["vertseg"]

    # Remove channel dimension
    im = np.squeeze(im, axis=0)
    subreg = np.squeeze(subreg, axis=0)
    vertseg = np.squeeze(vertseg, axis=0)

    return im, subreg, vertseg, transformed_data_dict["target"]


def pad_array_to_shape(arr, target_shape):  # noqa: ANN201
    """Pad an array to ``target_shape``, keeping the original centred.

    Args:
        arr: Array of shape ``(height, width, depth)``.
        target_shape: Target shape, which must be at least as large on every axis.

    Returns:
        The padded array and the padding offset that was applied.
    """
    # Calculate the padding needed for each dimension
    pad_h = (target_shape[0] - arr.shape[0]) // 2
    pad_w = (target_shape[1] - arr.shape[1]) // 2
    pad_d = (target_shape[2] - arr.shape[2]) // 2

    # Handle odd differences by adding an extra padding at the end if necessary
    pad_h2 = pad_h + (target_shape[0] - arr.shape[0]) % 2
    pad_w2 = pad_w + (target_shape[1] - arr.shape[1]) % 2
    pad_d2 = pad_d + (target_shape[2] - arr.shape[2]) % 2

    # Apply padding
    padded_arr = np.pad(arr, ((pad_h, pad_h2), (pad_w, pad_w2), (pad_d, pad_d2)), mode="constant")

    offset = (pad_h, pad_w, pad_d)

    return padded_arr, offset


def get_spine_poi(container, source: str) -> POI:
    """Find and load a subject's POI annotation file.

    Args:
        container: The subject's BIDS container.
        source: Value of the BIDS ``source-`` entity identifying which annotation
            set to read, i.e. the ``<source>`` in
            ``sub-..._seg-poi_source-<source>_poi.json``. This has to match the
            filenames on disk, so it depends on the dataset and has no default.

    Returns:
        The loaded POI.
    """
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format("poi")
    poi_query.filter("source", source)
    poi_candidate = poi_query.candidates[0]

    # poi = poi_candidate.open_ctd()
    poi = POI.load(poi_candidate.file["json"])

    return poi


def get_poi(container) -> POI:
    """Find and load a subject's POI annotation file, without extra filtering."""
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format("poi")
    if not poi_query.candidates:
        return None
    poi_candidate = poi_query.candidates[0]
    return str(poi_candidate.file["json"])


def get_ct(container, split=None) -> NII:
    """Find and load a subject's CT image, optionally restricted to one split."""
    ct_query = container.new_query(flatten=True)
    ct_query.filter_format("ct")
    ct_query.filter_filetype("nii.gz")  # only nifti files
    ct_query.filter("split", split)
    ct_candidate = ct_query.candidates[0]

    ct = ct_candidate.open_nii()
    return ct


def get_subreg(container, split=None) -> NII:
    """Find and load a subject's vertebra subregion mask, or None if it cannot be opened."""
    subreg_query = container.new_query(flatten=True)
    subreg_query.filter_format("msk")
    subreg_query.filter_filetype("nii.gz")  # only nifti files
    subreg_query.filter("seg", "subreg")
    subreg_query.filter("split", split)
    if not subreg_query.candidates:
        print("ERROR: No subreg candidates found!")
        return None
    subreg_candidate = subreg_query.candidates[0]

    try:
        subreg = subreg_candidate.open_nii()
    except Exception as e:
        print(f"Error opening subreg: {e!s}")
        return None
    else:
        return subreg


def get_vertseg(container) -> NII:
    """Find and load a subject's vertebra instance mask, or None if it cannot be opened."""
    vertseg_query = container.new_query(flatten=True)
    vertseg_query.filter_format("msk")
    vertseg_query.filter_filetype("nii.gz")  # only nifti files
    vertseg_query.filter("seg", "vert")
    if not vertseg_query.candidates:
        print("ERROR: No vertseg candidate found!")
        return None
    vertseg_candidate = vertseg_query.candidates[0]

    try:
        vertseg = vertseg_candidate.open_nii()
    except Exception as e:
        print(f"Error opening vertseg: {e!s}")
        return None
    else:
        return vertseg


def get_vertseg_bfile(container):  # noqa: ANN201
    """Return the BIDS file entry for a subject's vertebra instance mask."""
    vertseg_query = container.new_query(flatten=True)
    vertseg_query.filter_format("msk")
    vertseg_query.filter_filetype("nii.gz")  # only nifti files
    vertseg_query.filter("seg", "vert")
    if not vertseg_query.candidates:
        print("ERROR: No vertseg candidate found!")
        return None
    vertseg_candidates = vertseg_query.candidates
    return vertseg_candidates


def get_files(  # noqa: D103 - thin dispatcher over the getters above
    container,
    get_poi: Callable,
    get_ct: Callable,
    get_subreg: Callable,
    get_vertseg: Callable,
) -> tuple[POI, NII, NII, NII]:
    return (
        get_poi(container),
        get_ct(container),
        get_subreg(container),
        get_vertseg(container),
    )


def get_bounding_box(mask, vert, margin=5):  # noqa: ANN201
    """Get the bounding box of a given vertebra in a mask.

    Args:
        mask (numpy.ndarray): The mask to search for the vertex.
        vert (int): The vertebra to search for in the mask.
        margin (int, optional): The margin to add to the bounding box. Defaults to 2.

    Returns:
        tuple: A tuple containing the minimum and maximum values for the x, y, and z axes of the bounding box.
    """
    indices = np.where(mask == vert)
    x_min = np.min(indices[0]) - margin
    x_max = np.max(indices[0]) + margin
    y_min = np.min(indices[1]) - margin
    y_max = np.max(indices[1]) + margin
    z_min = np.min(indices[2]) - margin
    z_max = np.max(indices[2]) + margin

    # Make sure the bounding box is within the mask
    x_min = max(0, x_min)
    x_max = min(mask.shape[0], x_max)
    y_min = max(0, y_min)
    y_max = min(mask.shape[1], y_max)
    z_min = max(0, z_min)
    z_max = min(mask.shape[2], z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max


def process_container(  # noqa: ANN201
    subject,
    container,
    save_path: str | PathLike,
    rescale_zoom: tuple | None,
    get_files: Callable[[Subject_Container], tuple[POI, NII, NII, NII]],
):
    """Load one subject's image, masks and POIs, ready for cutout extraction."""
    poi, ct, subreg, vertseg = get_files(container)
    ct.reorient_(("L", "A", "S"))
    subreg.reorient_(("L", "A", "S"))
    vertseg.reorient_(("L", "A", "S"))
    # poi.reorient_centroids_to_(ct)
    poi.reorient_(axcodes_to=ct.orientation, _shape=ct.shape)  # the same as above? no reorient_centroids_to found in TPTBox

    vertebrae = {key[0] for key in poi.keys()}
    vertseg_arr = vertseg.get_array()

    summary = []
    for vert in vertebrae:
        if vert in vertseg_arr:
            x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(vertseg_arr, vert)
            ct_path = os.path.join(save_path, subject, str(vert), "ct.nii.gz")
            subreg_path = os.path.join(save_path, subject, str(vert), "subreg.nii.gz")
            vertseg_path = os.path.join(save_path, subject, str(vert), "vertseg.nii.gz")
            poi_path = os.path.join(save_path, subject, str(vert), "poi.json")
            # poi_det_path = os.path.join(save_path, subject, str(vert), 'poi_det.json')

            if not os.path.exists(os.path.join(save_path, subject, str(vert))):
                os.makedirs(os.path.join(save_path, subject, str(vert)))

            ct_cropped = ct.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))  # _slice(
            subreg_cropped = subreg.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))  # _slice(
            vertseg_cropped = vertseg.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))  # _slice(
            poi_cropped = poi.apply_crop(o_shift=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))  # crop_centroids(
            # poi_det.crop_centroids(o_shift = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))).save(poi_det_path)

            if rescale_zoom:
                ct_cropped.rescale_(rescale_zoom)
                subreg_cropped.rescale_(rescale_zoom)
                vertseg_cropped.rescale_(rescale_zoom)
                poi_cropped.rescale_(rescale_zoom)

            ct_cropped.save(ct_path, verbose=False)
            subreg_cropped.save(subreg_path, verbose=False)
            vertseg_cropped.save(vertseg_path, verbose=False)
            poi_cropped.save(poi_path, verbose=False)

            summary.append(
                {
                    "subject": subject,
                    "vertebra": vert,
                    "file_dir": os.path.join(save_path, subject, str(vert)),
                    # 'ct_nii_path': ct_path,
                    # 'subreg_nii_path': subreg_path,
                    # 'vertseg_nii_path': vertseg_path,
                    # 'poi_json_path': poi_path,
                }
            )

        else:
            print(f"Vertebra {vert} has no segmentation for subject {subject}")

    return summary
