"""Converting between landmark coordinates and the heatmaps the model predicts.

The coarse stage regresses one heatmap per landmark; decoding it back to a
coordinate is a coordinate expectation over the (already normalised) heatmap,
which keeps the result differentiable and sub-voxel accurate.
"""

import torch
from torch import nn


def coords_to_heatmaps(coords, target_shape, lambda_decay=1) -> torch.Tensor:
    """Render landmark coordinates as normalised exponential-decay heatmaps.

    Args:
        coords: Coordinates, ``(n_landmarks, 3)`` or ``(batch, n_landmarks, 3)``.
        target_shape: Spatial shape ``(height, width, depth)`` of each heatmap.
        lambda_decay: Decay length in voxels; larger spreads the peak further.

    Returns:
        Heatmaps summing to 1 over the spatial dimensions, batched to match ``coords``.
    """
    # Get the coords shape so we can handle batched and unbatched inputs
    unbatched = coords.dim() == 2
    if unbatched:
        coords = coords.unsqueeze(0)

    # Ensure sigma is a float for precision
    lambda_decay = float(lambda_decay)

    # Create a meshgrid for the target_shape: [K, L, M]
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.arange(target_shape[0], device=coords.device, dtype=torch.float),
        torch.arange(target_shape[1], device=coords.device, dtype=torch.float),
        torch.arange(target_shape[2], device=coords.device, dtype=torch.float),
        indexing="ij",
    )

    # Reshape grid for broadcasting: [1, K, L, M, 3] to match [B, N, 3]
    grid = torch.stack((grid_z, grid_y, grid_x), dim=-1).unsqueeze(0)

    # Expand coords_batch for broadcasting: [B, N, 3] -> [B, N, 1, 1, 1, 3] for matching grid's shape
    coords = coords.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)

    # Calculate squared distances: [B, N, K, L, M]
    dist_sq = ((grid - coords) ** 2).sum(dim=-1)

    # Compute the Gaussian heatmap: [B, N, K, L, M]
    heatmaps = torch.exp(-torch.sqrt(dist_sq) / lambda_decay)

    # Normalize the heatmaps
    heatmaps = heatmaps / heatmaps.sum(dim=(2, 3, 4), keepdim=True)

    # Squeeze out the batch dimension if unbatched
    if unbatched:
        heatmaps = heatmaps.squeeze(0)

    return heatmaps


def heatmaps_to_coords(pred_heatmaps) -> torch.Tensor:
    """Decode heatmaps to coordinates by taking their coordinate expectation.

    Args:
        pred_heatmaps (torch.Tensor): Batch of heatmaps of shape (batch_size, n_pois, height, width, depth).

    Returns:
        torch.Tensor: Batch of 3D coordinates of shape (batch_size, n_pois, 3).
    """
    device = pred_heatmaps.device
    unbatched = pred_heatmaps.dim() == 4

    if unbatched:
        pred_heatmaps = pred_heatmaps.unsqueeze(0)

    # Extract dimensions
    batch_size, n_pois, _, _, _ = pred_heatmaps.shape

    # Reshape pred_heatmaps and heatmap_coords for element-wise multiplication
    pred_heatmaps_reshaped = pred_heatmaps.view(batch_size, n_pois, -1)  # Shape: (batch_size, n_pois, H*W*D)
    heatmap_coords = create_coordinate_tensor(pred_heatmaps.shape[2:]).to(device)
    heatmap_coords_reshaped = heatmap_coords.view(3, -1)  # Shape: (3, H*W*D)

    heatmap_coords_reshaped = heatmap_coords_reshaped.to(device)

    # Element-wise multiplication and sum along the last dimension
    weighted_coords = torch.sum(pred_heatmaps_reshaped.unsqueeze(-1) * heatmap_coords_reshaped.t(), dim=2)  # Shape: (batch_size, n_pois, 3)

    # Reshape the result to (batch_size, n_pois, 3)
    coords = weighted_coords.view(batch_size, n_pois, 3)

    if unbatched:
        coords = coords.squeeze(0)

    return coords.to(device)


def create_coordinate_tensor(shape) -> torch.Tensor:
    """Build a coordinate grid.

    Args:
        shape: Spatial shape ``(height, width, depth)``.

    Returns:
        A ``(3, *shape)`` tensor whose channel ``i`` holds the index along axis ``i``.
    """
    # Generate a grid of coordinates along each dimension
    indices = [torch.arange(s, dtype=torch.float32) for s in shape]
    grid = torch.meshgrid(indices, indexing="ij")

    # Stack the coordinate grids along a new dimension to get the final coordinate tensor
    coords = torch.stack(grid, dim=0).float()

    return coords


class SoftArgmax3D(nn.Module):
    """Decode a normalised 3D heatmap to coordinates, differentiably.

    Takes the coordinate expectation over the heatmap. It does **not** apply a
    softmax first, so the input must already sum to 1 over its spatial dimensions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, heatmap) -> torch.Tensor:
        """Apply the soft-argmax operation on a 3D heatmap.

        The heatmap must already be a valid probability distribution: its elements
        must sum to 1 over the spatial dimensions.

        Args:
            heatmap (torch.Tensor): Input tensor of shape (b, n, h, w, d)

        Returns:
            torch.Tensor: Soft-argmax coordinates of shape (b, n, 3)
        """
        batch_size, num_maps, height, width, depth = heatmap.shape

        # Create coordinate grids for each dimension
        lin_h = torch.linspace(0, height - 1, steps=height, device=heatmap.device)
        lin_w = torch.linspace(0, width - 1, steps=width, device=heatmap.device)
        lin_d = torch.linspace(0, depth - 1, steps=depth, device=heatmap.device)

        # Expand grids to match batch size and number of maps
        grid_h = lin_h.view(1, 1, height, 1, 1).expand(batch_size, num_maps, -1, width, depth)
        grid_w = lin_w.view(1, 1, 1, width, 1).expand(batch_size, num_maps, height, -1, depth)
        grid_d = lin_d.view(1, 1, 1, 1, depth).expand(batch_size, num_maps, height, width, -1)

        # Compute the soft-argmax coordinates
        soft_argmax_h = torch.sum(heatmap * grid_h, dim=[2, 3, 4])
        soft_argmax_w = torch.sum(heatmap * grid_w, dim=[2, 3, 4])
        soft_argmax_d = torch.sum(heatmap * grid_d, dim=[2, 3, 4])

        # Stack results to get coordinates
        coords = torch.stack([soft_argmax_h, soft_argmax_w, soft_argmax_d], dim=-1)

        return coords
