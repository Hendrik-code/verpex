import pytorch_lightning as pl
import torch
from monai.networks.nets.regressor import Regressor
from torch import nn

from verpex.geometry.surface import surface_project_coords
from verpex.loss.loss_modules import get_loss_fn
from verpex.models.patch_extraction import PatchExtractor
from verpex.models.poi_transformer import FlexiblePoiTransformer


class RefinementModule(pl.LightningModule):
    """Defines a generic refinement module.

    The module is expected to have a forward method that takes a batch of data in
    dictionary format, containing the extracted features and the coarse predictions, and
    returns the batch. It needs to implement a calculate_loss method.
    """

    def __init__(self):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, batch) -> dict:
        """Refine ``batch`` in place and return it. Implemented by subclasses."""
        raise NotImplementedError

    def training_step(self, batch) -> torch.Tensor:
        """Run one training step and log the training metrics."""
        batch = self(batch)
        loss = self.calculate_loss(batch)
        metrics = self.calculate_metrics(batch, "train")
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch) -> torch.Tensor:
        """Run one validation step and log the validation metrics."""
        batch = self(batch)
        loss = self.calculate_loss(batch)
        metrics = self.calculate_metrics(batch, "val")
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return loss

    def calculate_loss(self, batch) -> torch.Tensor:
        """Return the scalar loss for ``batch``. Implemented by subclasses."""
        raise NotImplementedError

    def calculate_metrics(self, batch, mode) -> dict:
        """Return a metric dict for ``batch``, suffixed with ``mode``."""
        raise NotImplementedError

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return an Adam optimiser over this module's parameters."""
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class Identity(RefinementModule):
    """Defines a refinement module that does nothing.

    This is useful for testing the feature extraction module in isolation.
    """

    def __init__(
        self,
        project_gt: bool = False,
    ):
        super().__init__()
        self.project_gt = project_gt

    def forward(self, batch) -> dict:
        """Return ``batch`` unchanged."""
        return batch

    def calculate_loss(self, batch) -> torch.Tensor:  # noqa: ARG002
        """Return zero: there is nothing to train in this module."""
        return torch.zeros((), device=self.device)

    def calculate_metrics(self, batch, mode) -> dict:
        """Return coarse-stage metrics, so a no-refinement run is still measurable."""
        metrics = {}

        loss_mask = batch["loss_mask"]  # (batch_size, n_landmarks)
        fine_preds = batch["coarse_preds"]  # (batch_size, n_landmarks, 3)
        target = batch["target"]  # (batch_size, n_landmarks, 3)
        target_indices = batch["target_indices"]  # (batch_size, n_landmarks)

        if self.project_gt:
            # Project targets to surface
            surface = batch["surface"]
            target, projection_dist = surface_project_coords(target, surface)

            metrics[f"fine_projection_dist_{mode}"] = projection_dist.mean()

        # Consider zoom (mm per voxel), so these are comparable with every other
        # module's metrics - all of which report millimetres.
        zoom = batch["zoom"].to(target.device)
        zoom = zoom.unsqueeze(1)

        # Calculate the mean Euclidean distance between the predicted and target landmarks
        distances = torch.norm((fine_preds - target) * zoom, dim=-1)  # (batch_size, n_landmarks)
        distances_mean, distances_std = distances.mean(), distances.std()

        # Mask the distances with the loss mask
        distances_masked = distances[loss_mask]
        distances_masked_mean, distances_masked_std = (
            distances_masked.mean(),
            distances_masked.std(),
        )

        metrics[f"fine_mean_distance_{mode}"] = distances_mean
        metrics[f"fine_std_distance_{mode}"] = distances_std
        metrics[f"fine_mean_distance_masked_{mode}"] = distances_masked_mean
        metrics[f"fine_std_distance_masked_{mode}"] = distances_masked_std

        # Calculate the magnitude of the offsets. Identity applies none, so these
        # are zero - as tensors, to match every other value in the dict.
        zero = torch.zeros((), device=fine_preds.device)
        metrics[f"offsets_magnitude_mean_{mode}"] = zero
        metrics[f"offsets_magnitude_std_{mode}"] = zero
        metrics[f"offsets_magnitude_masked_mean_{mode}"] = zero
        metrics[f"offsets_magnitude_masked_std_{mode}"] = zero

        # Calculate mean Euclidian distance grouped by landmark type
        for landmark_type in target_indices.unique():
            landmark_mask = target_indices == landmark_type
            landmark_mask = landmark_mask * loss_mask
            distances_landmark = distances[landmark_mask]
            distances_landmark_mean = distances_landmark.mean()
            metrics[f"fine_mean_distance_{landmark_type.item()}_{mode}"] = distances_landmark_mean

        return metrics


class PatchTransformer(nn.Module):
    """Refine coarse landmark predictions with a transformer.

    Coarse per-landmark features are optionally concatenated with features extracted
    from an image patch around each coarse prediction, then a transformer predicts a
    per-landmark offset. ``refined_preds = coarse_preds + offsets``.

    The four ``use_*`` flags select which information the transformer sees. They
    replace what used to be seven separately maintained ablation subclasses; the old
    class names remain valid config ``type`` strings (see :data:`REFINEMENT_MODULES`)
    and map onto flag combinations.

    Args:
        n_landmarks: Number of landmark types, sizing the POI embedding.
        n_verts: Number of vertebra types, sizing the vertebra embedding.
        patch_size: Edge length of the cubic patch extracted around each coarse
            prediction. Ignored when ``use_patches`` is false.
        poi_feature_l: Width of the incoming coarse feature vector.
        patch_feature_l: Width of the feature vector the patch CNN produces.
        coord_embedding_l: Width of the coordinate embedding.
        poi_embedding_l: Width of the landmark-identity embedding.
        vert_embedding_l: Width of the vertebra-identity embedding.
        loss_fn: Name of the loss, resolved by :func:`verpex.loss.loss_modules.get_loss_fn`.
        project_gt: Project ground-truth landmarks onto the surface before computing loss.
        project_pred: Project refined predictions onto the surface after refinement.
        warmup_epochs: Blend ground truth into the coarse predictions for this many
            epochs. ``-1`` disables warmup.
        mlp_dim: Transformer feed-forward width.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        lr: Learning rate recorded for the optimiser configuration.
        patch_in_channels: Channel count of the input volume patches are taken from.
        use_poi_embedding: Give the transformer a landmark-identity embedding.
        use_vert_embedding: Give the transformer a vertebra-identity embedding.
        use_coarse_features: Feed the coarse feature vector to the transformer.
        use_patches: Extract and feed patch features.
        use_coarse_pred: Refine an existing coarse prediction. When false the
            transformer regresses coordinates directly and no patches are extracted.
    """

    def __init__(
        self,
        n_landmarks: int,
        loss_fn: str,
        n_verts: int = 22,
        patch_size: int = 16,
        poi_feature_l: int = 0,
        patch_feature_l: int = 0,
        coord_embedding_l: int = 0,
        poi_embedding_l: int = 0,
        vert_embedding_l: int = 0,
        project_gt: bool = False,
        project_pred: bool = False,
        warmup_epochs: int = -1,
        mlp_dim: int = 1024,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        lr: float = 1e-5,
        patch_in_channels: int = 1,
        use_poi_embedding: bool = True,
        use_vert_embedding: bool = True,
        use_coarse_features: bool = True,
        use_patches: bool = True,
        use_coarse_pred: bool = True,
    ):
        super().__init__()

        self.use_poi_embedding = use_poi_embedding
        self.use_vert_embedding = use_vert_embedding
        self.use_coarse_features = use_coarse_features
        self.use_patches = use_patches and use_coarse_pred
        self.use_coarse_pred = use_coarse_pred

        # Caught here rather than at the first forward pass, where it would surface as
        # an IndexError on an empty feature list part-way into a training run.
        if self.use_coarse_pred and not (self.use_coarse_features or self.use_patches):
            raise ValueError(
                "The transformer would receive no per-landmark features: set use_coarse_features or use_patches (or use_coarse_pred=False)."
            )
        if not self.use_coarse_pred and not self.use_coarse_features:
            # Without a coarse stage the coarse features are the transformer's only
            # per-landmark input. Disabling them silently dropped the tensor that
            # forward() passes anyway, leaving the embeddings as the sole input.
            raise ValueError("use_coarse_pred=False requires use_coarse_features=True: there is no other feature source.")

        if self.use_patches:
            self.patch_feature_extractor: PatchExtractor | None = PatchExtractor(
                patch_size=patch_size,
                feature_extraction_model=Regressor(
                    in_shape=(patch_in_channels, patch_size, patch_size, patch_size),
                    out_shape=(patch_feature_l,),
                    channels=(8, 16, 32),
                    strides=(2, 2, 2),
                    kernel_size=3,
                ),
            )
        else:
            self.patch_feature_extractor = None

        # Width of the token the transformer receives, before its own embeddings.
        feature_width = (poi_feature_l if self.use_coarse_features else 0) + (patch_feature_l if self.use_patches else 0)

        self.refinement_module = FlexiblePoiTransformer(
            poi_feature_l=feature_width,
            coord_embedding_l=coord_embedding_l if self.use_coarse_pred else None,
            poi_embedding_l=poi_embedding_l if self.use_poi_embedding else None,
            vert_embedding_l=vert_embedding_l if self.use_vert_embedding else None,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            n_landmarks=n_landmarks,
            n_verts=n_verts,
            dropout_rate=dropout,
        )

        self.loss_fn = get_loss_fn(loss_fn)
        self.project_gt = project_gt
        self.project_pred = project_pred
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        # Plain nn.Module, so there is no Lightning-provided current_epoch. The parent
        # PoiPredictionModule updates this each forward pass; without it the warmup
        # branch below raised AttributeError on the first training step.
        self.current_epoch = 0

    def forward(self, batch) -> dict:
        """Refine the coarse predictions in ``batch``, adding ``offsets`` and ``refined_preds``."""
        poi_indices = batch["poi_list_idx"] if self.use_poi_embedding else None
        vertebra_indices = batch["vert_list_idx"] if self.use_vert_embedding else None

        if not self.use_coarse_pred:
            # No coarse stage to refine: the transformer regresses coordinates directly.
            coords = self.refinement_module(
                coarse_preds=None,
                poi_indices=poi_indices,
                vertebra=vertebra_indices,
                poi_features=batch["coarse_features"],
            )
            batch["offsets"] = coords
            batch["refined_preds"] = coords
            return batch

        coarse_preds = batch["coarse_preds"]

        if self.warmup_epochs > 0 and self.training and self.current_epoch < self.warmup_epochs:
            # Blend ground truth into the coarse predictions during warmup.
            weight = self.current_epoch / self.warmup_epochs
            coarse_preds = weight * coarse_preds + (1 - weight) * batch["target"]

        # Keep float precision: PatchExtractor casts to long internally for indexing,
        # and the transformer accepts float coords for its coordinate embedding.
        # refined_preds = float_coarse + offsets preserves sub-voxel refinement.
        coarse_preds = coarse_preds.detach()

        features = []
        if self.use_coarse_features:
            # Detach coarse features so refiner gradients cannot propagate into the
            # DenseNet encoder. Prevents refiner blowup from corrupting the encoder
            # (observed as simultaneous coarse+refined divergence in earlier runs).
            features.append(batch["coarse_features"].detach())
        if self.patch_feature_extractor is not None:
            features.append(self.patch_feature_extractor(batch["input"], coarse_preds))
        poi_features = torch.cat(features, dim=-1) if len(features) > 1 else features[0]

        offsets = self.refinement_module(
            coarse_preds=coarse_preds,
            poi_indices=poi_indices,
            vertebra=vertebra_indices,
            poi_features=poi_features,
        )

        batch["offsets"] = offsets
        batch["refined_preds"] = coarse_preds + offsets
        if self.project_pred:
            batch["refined_preds"], _ = surface_project_coords(batch["refined_preds"], batch["surface"])

        return batch

    def calculate_loss(self, batch) -> torch.Tensor:
        """Return the refinement loss, computed in millimetres."""
        target = batch["target"]
        surface = batch["surface"]
        if self.project_gt:
            target, _ = surface_project_coords(target, surface)

        # Scale to millimetres so the loss is comparable across resolutions.
        zoom = batch["zoom"].to(target.device).unsqueeze(1)
        return self.loss_fn(
            batch["refined_preds"] * zoom,
            target * zoom,
            batch["loss_mask"],
            surface,
            # Set by PoiNeighborPredictionModule to weight the current vertebra
            # against its neighbours; absent for single-vertebra batches.
            batch.get("poi_loss_weights"),
        )

    def calculate_metrics(self, batch, mode) -> dict:
        """Return refinement metrics for ``batch``.

        Distances are reported in millimetres, and separately over all landmarks and
        over only those the loss mask selects.

        Args:
            batch: A batch already passed through :meth:`forward`.
            mode: ``"train"`` or ``"val"``, used to suffix every metric name.

        Returns:
            A mapping of metric name to scalar tensor.
        """
        metrics = {}

        loss_mask = batch["loss_mask"]  # (batch_size, n_landmarks)
        fine_preds = batch["refined_preds"]  # (batch_size, n_landmarks, 3)
        target = batch["target"]  # (batch_size, n_landmarks, 3)
        target_indices = batch["target_indices"]  # (batch_size, n_landmarks)

        if self.project_gt:
            target = batch["target"]
            # Project targets to surface
            surface = batch["surface"]
            target, projection_dist = surface_project_coords(target, surface)

            metrics[f"fine_projection_dist_{mode}"] = projection_dist.mean()

        # Consider zoom (mm per voxel)
        zoom = batch["zoom"].to(target.device)
        zoom = zoom.unsqueeze(1)

        # Calculate the mean Euclidean distance between the predicted and target landmarks
        distances = torch.norm((fine_preds - target) * zoom, dim=-1)  # (batch_size, n_landmarks)
        distances_mean, distances_std = distances.mean(), distances.std()

        # Mask the distances with the loss mask
        distances_masked = distances[loss_mask]
        distances_masked_mean, distances_masked_std = (
            distances_masked.mean(),
            distances_masked.std(),
        )

        metrics[f"fine_mean_distance_{mode}"] = distances_mean
        metrics[f"fine_std_distance_{mode}"] = distances_std
        metrics[f"fine_mean_distance_masked_{mode}"] = distances_masked_mean
        metrics[f"fine_std_distance_masked_{mode}"] = distances_masked_std

        # Calculate the magnitude of the offsets
        offsets = batch["offsets"]
        offsets_magnitude = torch.norm(offsets, dim=-1)
        offsets_magnitude_mean, offsets_magnitude_std = (
            offsets_magnitude.mean(),
            offsets_magnitude.std(),
        )

        # Mask the offsets with the loss mask
        offsets_masked = offsets_magnitude * loss_mask
        offsets_masked_mean, offsets_masked_std = (
            offsets_masked.mean(),
            offsets_masked.std(),
        )

        metrics[f"offsets_magnitude_mean_{mode}"] = offsets_magnitude_mean
        metrics[f"offsets_magnitude_std_{mode}"] = offsets_magnitude_std
        metrics[f"offsets_magnitude_masked_mean_{mode}"] = offsets_masked_mean
        metrics[f"offsets_magnitude_masked_std_{mode}"] = offsets_masked_std

        # Calculate mean Euclidian distance grouped by landmark type
        for landmark_type in target_indices.unique():
            landmark_mask = target_indices == landmark_type
            landmark_mask = landmark_mask * loss_mask
            distances_landmark = distances[landmark_mask]
            distances_landmark_mean = distances_landmark.mean()
            metrics[f"fine_mean_distance_{landmark_type.item()}_{mode}"] = distances_landmark_mean

        return metrics

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return an Adam optimiser over this module's parameters."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def _variant(**flags):
    """Return a factory for a :class:`PatchTransformer` with fixed ablation flags.

    Each historical ablation subclass becomes one flag combination, so a config's
    ``"type"`` string keeps selecting the same architecture it always did.

    The flags a variant fixes are what its name *means*, so a config that also sets
    one of them is contradicting itself; that raises rather than silently building a
    model that does not match the type string recorded in its checkpoint.
    """

    def factory(**kwargs):
        conflicting = sorted(set(flags) & set(kwargs))
        if conflicting:
            raise ValueError(
                f"This refinement type already fixes {', '.join(conflicting)}; "
                f"remove it from the config, or use type 'PatchTransformer' to set the flags yourself."
            )
        return PatchTransformer(**flags, **kwargs)

    return factory


#: Config ``"type"`` string -> refinement module.
#: Consumed by :func:`verpex.modules.poi_module.create_refinement_module`.
#:
#: The ``No*`` entries were separate classes until they were merged into
#: :class:`PatchTransformer`; they remain valid config values.
REFINEMENT_MODULES = {
    "Identity": Identity,
    "PatchTransformer": PatchTransformer,
    "NoPoiPatchTransformer": _variant(use_poi_embedding=False),
    "NoVertPatchTransformer": _variant(use_vert_embedding=False),
    "NoPoiVertPatchTransformer": _variant(use_poi_embedding=False, use_vert_embedding=False),
    "NoPoiFeaturePatchTransformer": _variant(use_coarse_features=False),
    "NoPoiVertFeaturePatchTransformer": _variant(use_poi_embedding=False, use_vert_embedding=False, use_coarse_features=False),
    "NoCoarsePredTransformer": _variant(use_coarse_pred=False, use_patches=False),
    "FeatureTransformer": _variant(use_patches=False),
}
