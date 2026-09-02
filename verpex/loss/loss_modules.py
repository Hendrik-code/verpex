"""Loss functions for landmark regression.

All losses share the signature ``(pred, target, mask=None, surface=None,
weights=None)`` so they are interchangeable in a config, and so
:class:`CompoundLoss` can combine any of them. ``pred`` and ``target`` are
``(batch, n_landmarks, 3)`` coordinate tensors; ``mask`` is a
``(batch, n_landmarks)`` boolean selecting the landmarks that count; ``weights`` is
an optional ``(batch, n_landmarks)`` tensor giving each landmark a relative weight
in the mean, used by the neighbour-aware module to emphasise the current vertebra
over its neighbours.

Callers pass coordinates already scaled to millimetres, so a loss value means the
same thing regardless of the voxel spacing the sample was acquired at.
"""

import torch
from torch import nn

from verpex.geometry.surface import surface_project_coords


def masked_weighted_mean(values: torch.Tensor, mask=None, weights=None) -> torch.Tensor:
    """Reduce per-landmark values to a scalar, honouring a mask and optional weights.

    Args:
        values: Per-landmark values, ``(batch, n_landmarks)`` or
            ``(batch, n_landmarks, 3)``. A trailing coordinate axis is averaged first,
            so each landmark contributes once regardless of dimensionality.
        mask: Boolean ``(batch, n_landmarks)`` selecting landmarks that count.
        weights: Optional ``(batch, n_landmarks)`` relative weights.

    Returns:
        A scalar tensor. Zero when nothing is selected - a plain ``.mean()`` over an
        empty selection returns NaN, which would poison the whole batch's loss. That
        happens routinely here: a vertebra at the end of the spine has no neighbour,
        so its entire block is masked out.
    """
    if weights is None:
        # Reduce in one step over the original shape, so an unweighted call is
        # bit-identical to a plain `values[mask].mean()`. Pre-averaging the
        # coordinate axis first is mathematically equal but reassociates the
        # float sum, which would perturb every existing run in the last digits.
        selected = values if mask is None else values[mask]
        return selected.mean() if selected.numel() else values.sum() * 0.0

    if values.dim() == 3:
        values = values.mean(dim=-1)

    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0  # keeps the graph connected, unlike torch.tensor(0.)

    selected_weights = weights.expand_as(values)[mask].to(values.dtype)
    total = selected_weights.sum()
    if total == 0:
        return values.sum() * 0.0
    return (selected * selected_weights).sum() / total


class SurfaceDistanceLoss(nn.Module):
    """Penalise predictions that lie off the vertebra surface.

    Ignores ``target`` entirely: it measures only how far each prediction sits from
    the surface mesh, which is useful as one term of a :class:`CompoundLoss`
    alongside a term that does compare against the ground truth.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None, surface=None, weights=None) -> torch.Tensor:  # noqa: ARG002
        """Return the mean distance from the predictions to the surface.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Unused; present so every loss shares one signature.
            mask: Boolean ``(batch, n_landmarks)`` selecting the landmarks to score.
            weights: Optional ``(batch, n_landmarks)`` relative weights.
            surface: Surface point cloud to project onto. When ``None`` the loss is
                zero, so a config can enable this term only where surfaces exist.

        Returns:
            A scalar loss tensor.
        """
        if surface is None:
            return torch.tensor(0.0, device=pred.device)

        dist_to_surface = surface_project_coords(pred, surface)[1]  # (batch, n_landmarks)
        # Scaled down by 10 to keep this term commensurate with the coordinate
        # losses it is usually compounded with.
        return masked_weighted_mean(dist_to_surface, mask, weights) / 10


class WingLoss3D(nn.Module):
    """Wing loss, which weights small landmark errors more heavily than L1.

    Logarithmic inside ``omega`` and linear outside it, so sub-``omega`` refinement
    keeps producing gradient instead of flattening out the way L1 does.

    Args:
        omega: Error magnitude at which the loss switches from log to linear.
        epsilon: Curvature of the logarithmic part; smaller is steeper.
    """

    def __init__(self, omega=5, epsilon=2):
        super().__init__()
        self.omega = torch.tensor(omega, dtype=torch.float32)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32)

    def forward(self, pred, target, mask=None, surface=None, weights=None) -> torch.Tensor:  # noqa: ARG002
        """Return the mean wing loss over the selected landmarks.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Ground-truth coordinates, same shape.
            mask: Boolean ``(batch, n_landmarks)`` selecting landmarks to score.
            weights: Optional ``(batch, n_landmarks)`` relative weights.
            surface: Unused; present so every loss shares one signature.
            weights: Optional ``(batch, n_landmarks)`` relative weights.

        Returns:
            A scalar loss tensor.
        """
        delta_y = torch.abs(pred - target)

        small_mask = delta_y < self.omega
        loss_small = self.omega * torch.log(1 + delta_y / self.epsilon) * small_mask

        # Offset chosen so the linear part meets the logarithmic part at omega.
        large_mask = delta_y >= self.omega
        c = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon)
        loss_large = (delta_y - c) * large_mask

        loss = loss_small + loss_large
        return masked_weighted_mean(loss, mask, weights)


class L1LossMasked(nn.Module):
    """Mean absolute coordinate error over the landmarks the mask selects."""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask=None, surface=None, weights=None) -> torch.Tensor:  # noqa: ARG002
        """Return the mean L1 error.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Ground-truth coordinates, same shape.
            mask: Boolean ``(batch, n_landmarks)``. When ``None`` every landmark counts.
            weights: Optional ``(batch, n_landmarks)`` relative weights.
            surface: Unused; present so every loss shares one signature.

        Returns:
            A scalar loss tensor.
        """
        loss = self.l1_loss(pred, target)
        return masked_weighted_mean(loss, mask, weights)


class L2LossMasked(nn.Module):
    """Mean squared coordinate error over the landmarks the mask selects."""

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask=None, surface=None, weights=None) -> torch.Tensor:  # noqa: ARG002
        """Return the mean squared error.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Ground-truth coordinates, same shape.
            mask: Boolean ``(batch, n_landmarks)``. When ``None`` every landmark counts.
            weights: Optional ``(batch, n_landmarks)`` relative weights.
            surface: Unused; present so every loss shares one signature.

        Returns:
            A scalar loss tensor.
        """
        loss = self.mse_loss(pred, target)
        return masked_weighted_mean(loss, mask, weights)


class CompoundLoss(nn.Module):
    """Weighted sum of several losses.

    Args:
        loss_fns: The losses to combine.
        weights: One weight per loss, summing to 1 (within 1e-6). Defaults to equal
            weights.

    Raises:
        ValueError: If explicit weights do not sum to 1.
    """

    def __init__(self, loss_fns, weights=None):
        super().__init__()
        self.loss_fns = loss_fns
        if weights is None:
            weights = [1.0 / len(loss_fns)] * len(loss_fns)
        elif abs(sum(weights) - 1.0) > 1e-6:
            # An exact `== 1.0` rejected ordinary inputs: sum([0.7, 0.2, 0.1]) is
            # 0.9999999999999999 in binary floating point.
            raise ValueError(f"CompoundLoss weights must sum to 1.0, got {sum(weights)}.")
        self.weights = weights

    def forward(self, pred, target, mask=None, surface=None, weights=None) -> torch.Tensor:
        """Return the weighted sum of the component losses.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Ground-truth coordinates, same shape.
            mask: Boolean ``(batch, n_landmarks)`` passed to each component.
            weights: Optional ``(batch, n_landmarks)`` relative weights.
            surface: Surface point cloud passed to each component.

        Returns:
            A scalar loss tensor.
        """
        total_loss = torch.zeros((), device=pred.device)
        for loss_fn, weight in zip(self.loss_fns, self.weights):
            total_loss = total_loss + weight * loss_fn(pred, target, mask, surface, weights)
        return total_loss


def get_loss_fn(loss_fn: str | list[str]) -> nn.Module:
    """Resolve a loss by name, as written in an experiment config.

    Args:
        loss_fn: One of ``"L1"``, ``"L2"``, ``"WingLoss"`` or ``"SD"``, or a list of
            them, which builds an equally weighted :class:`CompoundLoss`.

    Returns:
        The loss module.

    Raises:
        ValueError: If the name is not recognised.
    """
    if isinstance(loss_fn, list):
        n = len(loss_fn)
        return CompoundLoss([get_loss_fn(lf) for lf in loss_fn], weights=[1.0 / n] * n)
    if loss_fn == "L1":
        return L1LossMasked()
    if loss_fn == "WingLoss":
        return WingLoss3D()
    if loss_fn == "L2":
        return L2LossMasked()
    if loss_fn == "SD":
        return SurfaceDistanceLoss()
    raise ValueError(f"Unknown loss function: {loss_fn}")
