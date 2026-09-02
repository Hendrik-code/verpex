import torch
from torch import nn


class PatchExtractor(nn.Module):
    """Extract per-landmark patch features from a feature volume.

    A patch is cut around each landmark's coarse prediction and encoded by a small
    CNN, giving the refinement transformer local image evidence to work from.
    """

    def __init__(self, patch_size, feature_extraction_model):
        super().__init__()

        self.patch_size = patch_size

        self.feature_extraction_model = feature_extraction_model

    def forward(self, x, centroids) -> torch.Tensor:
        """Extract a patch around each centroid and encode it.

        Args:
            x: Feature volume, ``(batch, channels, height, width, depth)``.
            centroids: Patch centres, ``(batch, n_landmarks, 3)``.

        Returns:
            Per-landmark patch features, ``(batch, n_landmarks, feature_l)``.
        """
        # Extract the patches
        patches = self.extract_patches(x, centroids)  # (B, N, C, patch_size, patch_size, patch_size)

        # Pass the patches through 3 conv layers
        B, N, C, P, _, _ = patches.shape
        patches = patches.view(B * N, C, P, P, P)  # Batch flatten
        out = self.feature_extraction_model(patches)  # (B*N, out_channels)
        out = out.view(B, N, -1)  # (B, N, out_channels)

        return out

    def extract_patches(self, vol, centroids) -> torch.Tensor:
        """Extract 3D patches, zero-padding where a patch runs off the volume.

        Args:
            vol: Input volume, ``(batch, channels, height, width, depth)``.
            centroids: Patch centres, ``(batch, n_landmarks, 3)``. Float or int;
                rounded to long for indexing.

        Returns:
            Patches, ``(batch, n_landmarks, channels, P, P, P)`` with ``P = patch_size``.
        """
        patch_size = self.patch_size
        B, _C, H, W, D = vol.shape
        N = centroids.shape[1]
        device = vol.device

        # Round centroids to integer indices; accept float or int input.
        # round(), not a bare .long(): truncation biases every patch up to a voxel
        # toward the origin relative to the sub-voxel coarse prediction.
        centroids = centroids.round().long()

        # 1D offset grid of length P centred on 0 → positions [-P//2, ..., P//2 - 1]
        offs = torch.arange(patch_size, device=device) - patch_size // 2

        # Per-axis coord arrays: (B, N, P) via broadcasting the offset over the centroid
        cx = centroids[..., 0:1] + offs.view(1, 1, patch_size)  # x indexes H dim
        cy = centroids[..., 1:2] + offs.view(1, 1, patch_size)  # y indexes W dim
        cz = centroids[..., 2:3] + offs.view(1, 1, patch_size)  # z indexes D dim

        # Broadcast to full 3-D grid (B, N, P, P, P)
        cx_f = cx.unsqueeze(-1).unsqueeze(-1).expand(B, N, patch_size, patch_size, patch_size)
        cy_f = cy.unsqueeze(-2).unsqueeze(-1).expand(B, N, patch_size, patch_size, patch_size)
        cz_f = cz.unsqueeze(-2).unsqueeze(-2).expand(B, N, patch_size, patch_size, patch_size)

        # Validity mask: True where the coord is inside the volume
        valid = (cx_f >= 0) & (cx_f < H) & (cy_f >= 0) & (cy_f < W) & (cz_f >= 0) & (cz_f < D)  # (B, N, P, P, P)

        # Clamp coords so gather is safe; masked-out positions are zeroed later
        cx_c = cx_f.clamp(0, H - 1)
        cy_c = cy_f.clamp(0, W - 1)
        cz_c = cz_f.clamp(0, D - 1)

        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1, 1).expand(B, N, patch_size, patch_size, patch_size)

        # Advanced indexing across the batch and 3 spatial dims, keeping C as a slice.
        # Because the slice (:) sits between fancy indices, the fancy dims are moved to
        # the front → result shape (B, N, P, P, P, C).
        patches = vol[b_idx, :, cx_c, cy_c, cz_c]

        # Rearrange to (B, N, C, P, P, P) and zero out invalid positions.
        # Cast to float32 to match model weights (dataset may hand us float64).
        patches = patches.permute(0, 1, 5, 2, 3, 4).contiguous().float()
        patches = patches * valid.unsqueeze(2)

        return patches
