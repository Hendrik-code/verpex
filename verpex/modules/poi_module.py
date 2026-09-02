"""The top-level POI prediction module.

Composes a coarse feature-extraction stage with a refinement stage and trains
them jointly, optionally freezing the coarse stage once its validation loss has
plateaued.
It also includes helper functions for creating feature extraction and refinement modules.

Classes:
    - PoiPredictionModule: A PyTorch Lightning module for predicting points of interest.

Functions:
    - create_feature_extraction_module: Creates a feature extraction based on a given configuration.
    - create_refinement_module: Creates a refinement module based on the given configuration.
"""

from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import LRSchedulerConfig, OptimizerLRScheduler
from torch import nn

from verpex.modules.feature_extraction import FEATURE_EXTRACTION_MODULES
from verpex.modules.refinement import REFINEMENT_MODULES
from verpex.registry import build


class PoiPredictionModule(pl.LightningModule):
    """A PyTorch Lightning module for POI prediction.

    coarse_config (dict): Configuration for the coarse feature extraction module.
        refinement_config (dict): Configuration for the refinement module.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        loss_weights (list, optional): Weights for the feature extraction and refinement losses. Defaults to None.
        optimizer (str, optional): Optimizer algorithm. Defaults to "AdamW".
        scheduler_config (dict, optional): Configuration for the learning rate scheduler. Defaults to None.
        feature_freeze_patience (int, optional): Number of epochs without improvement before freezing the feature extraction module. Defaults to None.

    Attributes:
        feature_extraction_module (Module): The feature extraction module.
        refinement_module (Module): The refinement module.
        lr (float): Learning rate for the optimizer.
        loss_weights (Tensor): Weights for the feature extraction and refinement losses.
        feature_freeze_patience (int): Number of epochs without improvement before freezing the feature extraction module.
        best_feature_loss (float): Best feature loss achieved during validation.
        val_feature_loss_outputs (list): List of feature loss values during validation.
        epochs_without_improvement (int): Number of epochs without improvement during validation.
        feature_extactor_frozen (bool): Flag indicating if the feature extraction module is frozen.
        optimizer (str): Optimizer algorithm.
        scheduler_config (dict): Configuration for the learning rate scheduler.

    Methods:
        forward(*args, **kwargs): Forward pass of the module.
        training_step(*args, **kwargs): Training step of the module.
        validation_step(*args, **kwargs): Validation step of the module.
        on_validation_epoch_end(): Callback function called at the end of each validation epoch.
        configure_optimizers(): Configures the optimizer and learning rate scheduler.
        calculate_metrics(batch, mode): Calculates metrics for the given batch and mode.
        freeze_feature_extractor(): Freezes the feature extraction module.
    """

    def __init__(
        self,
        coarse_config,
        refinement_config,
        lr=1e-4,
        loss_weights=None,
        optimizer="AdamW",
        scheduler_config=None,
        feature_freeze_patience=None,
        weight_decay=1e-4,
    ):
        super().__init__()
        if loss_weights is None:
            loss_weights = [1, 1]
        self.feature_extraction_module = create_feature_extraction_module(coarse_config)
        self.refinement_module = create_refinement_module(refinement_config)
        self.lr = lr
        # Per-submodule LRs (fall back to the module-level lr). Honouring these
        # lets configs use a smaller LR for the transformer refiner (standard for
        # coarse-to-fine models — the refiner needs a gentler step or it
        # diverges and drags the coarse encoder with it).
        self.coarse_lr = coarse_config.get("params", {}).get("lr", lr)
        self.refiner_lr = refinement_config.get("params", {}).get("lr", lr)
        self.weight_decay = weight_decay
        self.loss_weights = torch.tensor(loss_weights) / torch.sum(torch.tensor(loss_weights))
        self.feature_freeze_patience = feature_freeze_patience
        self.best_feature_loss = np.inf
        self.val_feature_loss_outputs = []
        self.epochs_without_improvement = 0
        self.feature_extactor_frozen = False
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, *args, **kwargs) -> dict:
        """Performs the forward pass of the module.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The processed batch after passing through the feature extraction and refinement modules.

        Raises:
            ValueError: If batch input is not provided.
        """
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")

        batch = self.feature_extraction_module(batch)
        # A plain nn.Module refiner (e.g. PatchTransformer) gets no epoch counter
        # from Lightning but needs one for its warmup schedule. A LightningModule
        # refiner (e.g. Identity) already owns `current_epoch` as a read-only
        # property, so assigning to it raises AttributeError - skip those.
        if not isinstance(self.refinement_module, pl.LightningModule):
            self.refinement_module.current_epoch = self.current_epoch
        batch = self.refinement_module(batch)
        return batch

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        """Run one training step over the coarse and refinement stages.

        Args:
            *args: The batch, positionally.
            **kwargs: The batch, as ``batch=``.

        Returns:
            The combined weighted loss.

        Raises:
            ValueError: If no batch was passed.
        """
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")
        batch = self(batch)

        # Calculate the feature extraction loss
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        # Calculate the refinement loss
        refinement_loss = self.refinement_module.calculate_loss(batch)
        loss = feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]

        metrics = self.calculate_metrics(batch, "train")
        batch_size = batch["input"].shape[0]

        self.log("train_loss", loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        """Run one validation step over the coarse and refinement stages.

        Args:
            *args: The batch, positionally.
            **kwargs: The batch, as ``batch=``.

        Returns:
            The combined weighted loss.

        Raises:
            ValueError: If no batch was passed.
        """
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")
        batch = self(batch)

        # Calculate the feature extraction loss
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        # Calculate the refinement loss
        refinement_loss = self.refinement_module.calculate_loss(batch)
        loss = feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]

        metrics = self.calculate_metrics(batch, "val")
        batch_size = batch["input"].shape[0]

        self.val_feature_loss_outputs.append(feature_loss)

        self.log("val_feature_loss", feature_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_refinement_loss", refinement_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Freeze the coarse stage once its validation loss stops improving."""
        # Check if the feature extraction module should be frozen
        if self.feature_extactor_frozen:
            return

        avg_feature_loss = torch.stack(self.val_feature_loss_outputs).mean()
        self.val_feature_loss_outputs.clear()

        if self.feature_freeze_patience is not None:
            if avg_feature_loss < self.best_feature_loss:
                self.best_feature_loss = avg_feature_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.feature_freeze_patience and not self.feature_extactor_frozen:
                    self.freeze_feature_extractor()
                    self.feature_extactor_frozen = True
                    print("Feature extraction module frozen")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build the optimiser and scheduler named in the config.

        The coarse and refinement stages get separate parameter groups, so each can
        run at its own learning rate.
        """
        optimizer_class = getattr(torch.optim, self.optimizer)
        param_groups = [
            {
                "params": self.feature_extraction_module.parameters(),
                "lr": self.coarse_lr,
                "name": "coarse",
            },
            {
                "params": self.refinement_module.parameters(),
                "lr": self.refiner_lr,
                "name": "refiner",
            },
        ]
        optimizer = optimizer_class(param_groups, lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler_config:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler_config["type"])
            scheduler = scheduler_class(optimizer, **self.scheduler_config["params"])

            scheduler_config: dict[str, Any] = {"scheduler": scheduler, "interval": "epoch"}
            if "monitor" in self.scheduler_config:
                scheduler_config["monitor"] = self.scheduler_config["monitor"]

            return [optimizer], [cast("LRSchedulerConfig", scheduler_config)]

        return optimizer

    def calculate_metrics(self, batch, mode) -> dict:
        """Calculates metrics for the given batch and mode.

        Parameters:
            batch (Tensor): The input batch.
            mode (str): The mode of calculation.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        feature_metrics = self.feature_extraction_module.calculate_metrics(batch, mode)
        refinement_metrics = self.refinement_module.calculate_metrics(batch, mode)

        return {**feature_metrics, **refinement_metrics}

    def freeze_feature_extractor(self) -> None:
        """Freeze the coarse stage so training only updates the refiner.

        Sets ``requires_grad = False`` on every feature-extractor parameter.
        """
        self.log("feature_frozen", True, on_epoch=True, sync_dist=True)
        for param in self.feature_extraction_module.parameters():
            param.requires_grad = False


def create_feature_extraction_module(config) -> nn.Module:
    """Create a feature extraction module based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the module.

    Returns:
        module_type: An instance of the feature extraction module.

    Raises:
        UnknownTypeError: If the config names a module that is not registered.
    """
    return build(FEATURE_EXTRACTION_MODULES, "feature extraction module", config)


def create_refinement_module(config) -> nn.Module:
    """Create a refinement module based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the module.

    Returns:
        object: An instance of the refinement module.

    Raises:
        UnknownTypeError: If the config names a module that is not registered.

    Example:
        config = {
            "type": "SomeModule",
            "params": {
                "param1": value1,
                "param2": value2
            }
        }
        module = create_refinement_module(config)
    """
    return build(REFINEMENT_MODULES, "refinement module", config)


class PoiNeighborPredictionModule(PoiPredictionModule):
    """Multi-vertebrae POI prediction module that extends PoiPredictionModule.

    Predicts landmarks for a vertebra and its two neighbours at once, weighting the
    current vertebra's landmarks against its neighbours' in the loss. The neighbour
    datasets concatenate one equally sized landmark block per vertebra, block 0 being
    the current one.

    The weights are applied as per-landmark weights in a single loss call rather than
    by slicing the batch per vertebra, so an absent neighbour (a vertebra at either
    end of the spine, or one dropped by ``neighbor_drop_prob``) contributes nothing
    instead of producing NaN from a mean over an empty selection.
    """

    def __init__(
        self,
        coarse_config,
        refinement_config,
        lr=1e-4,
        loss_weights=None,
        optimizer="AdamW",
        scheduler_config=None,
        feature_freeze_patience=None,
        current_weight=1.0,
        neighbor_weight=0.2,
        weight_decay=1e-4,
    ):
        """Initialise a neighbour-aware POI prediction module.

        Args:
            current_weight: Loss weight for the current vertebra's landmarks.
            neighbor_weight: Loss weight for the neighbouring vertebrae's landmarks.
            **kwargs: Forwarded to :class:`PoiPredictionModule`.
        """
        super().__init__(
            coarse_config=coarse_config,
            refinement_config=refinement_config,
            lr=lr,
            loss_weights=loss_weights,
            optimizer=optimizer,
            scheduler_config=scheduler_config,
            feature_freeze_patience=feature_freeze_patience,
            weight_decay=weight_decay,
        )

        if current_weight < 0 or neighbor_weight < 0:
            raise ValueError(f"Loss weights must be non-negative, got current={current_weight}, neighbor={neighbor_weight}.")
        self.current_weight = current_weight
        self.neighbor_weight = neighbor_weight

        # Update hyperparameters to include new parameters
        self.save_hyperparameters()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        """Override training step to use multi-vertebrae loss calculation."""
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")

        batch = self(batch)

        # Use multi-vertebrae loss calculation
        loss = self._calculate_multi_vertebrae_loss(batch)

        metrics = self.calculate_metrics(batch, "train")
        batch_size = batch["input"].shape[0]

        self.log("train_loss", loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        """Override validation step to use multi-vertebrae loss calculation."""
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")

        batch = self(batch)

        # Use multi-vertebrae loss calculation
        loss = self._calculate_multi_vertebrae_loss(batch)

        # Also calculate component losses for logging
        feature_loss = self._calculate_feature_loss_component(batch)
        refinement_loss = self._calculate_refinement_loss_component(batch)

        metrics = self.calculate_metrics(batch, "val")
        batch_size = batch["input"].shape[0]

        self.val_feature_loss_outputs.append(feature_loss)

        self.log("val_feature_loss", feature_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_refinement_loss", refinement_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)

        return loss

    def _landmark_weights(self, batch):
        """Return per-landmark loss weights, or None for a single-vertebra batch.

        The neighbour datasets concatenate one equally sized block of landmarks per
        vertebra, block 0 being the current vertebra. This turns that layout into a
        ``(batch, n_landmarks)`` weight tensor so the whole loss can be computed in
        one call, rather than slicing the batch into blocks and calling each loss
        once per block per sample. That matters: with ``project_gt`` enabled the
        per-block form would run one surface projection per block per sample instead
        of one per batch.

        Args:
            batch: A batch, after the forward pass.

        Returns:
            A ``(batch, n_landmarks)`` weight tensor, or ``None`` when the batch is
            not a multi-vertebra batch or the weights are uniform anyway.
        """
        if "n_vertebrae" not in batch or "n_pois_per_vertebra" not in batch:
            return None
        if self.current_weight == self.neighbor_weight:
            return None  # uniform weights change nothing; keep the cheaper path

        target = batch["target"]
        batch_size, n_landmarks = target.shape[0], target.shape[1]

        def _scalar(key):
            value = batch[key]
            return int(value) if isinstance(value, int) else int(value[0])

        n_vertebrae, n_per_vertebra = _scalar("n_vertebrae"), _scalar("n_pois_per_vertebra")
        if n_vertebrae * n_per_vertebra != n_landmarks:
            raise ValueError(
                f"Batch declares {n_vertebrae} vertebrae x {n_per_vertebra} landmarks "
                f"= {n_vertebrae * n_per_vertebra}, but carries {n_landmarks} landmarks."
            )

        weights = torch.full((n_landmarks,), float(self.neighbor_weight), device=target.device)
        weights[:n_per_vertebra] = float(self.current_weight)  # block 0 is the current vertebra
        return weights.unsqueeze(0).expand(batch_size, n_landmarks)

    def _calculate_multi_vertebrae_loss(self, batch):
        """Return the loss, weighting the current vertebra against its neighbours.

        Falls back to the flat loss when the batch carries no per-vertebra layout.
        """
        weights = self._landmark_weights(batch)
        if weights is None:
            return self._calculate_standard_loss(batch)

        batch = {**batch, "poi_loss_weights": weights}
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        refinement_loss = self.refinement_module.calculate_loss(batch)
        return feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]

    def _calculate_feature_loss_component(self, batch):
        """Return the coarse-stage loss alone, with the same weighting, for logging."""
        weights = self._landmark_weights(batch)
        if weights is None:
            return self.feature_extraction_module.calculate_loss(batch)
        return self.feature_extraction_module.calculate_loss({**batch, "poi_loss_weights": weights})

    def _calculate_refinement_loss_component(self, batch):
        """Return the refinement loss alone, with the same weighting, for logging."""
        weights = self._landmark_weights(batch)
        if weights is None:
            return self.refinement_module.calculate_loss(batch)
        return self.refinement_module.calculate_loss({**batch, "poi_loss_weights": weights})

    def _calculate_standard_loss(self, batch):
        """Fallback to standard loss calculation for single-vertebra batches."""
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        refinement_loss = self.refinement_module.calculate_loss(batch)
        return feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]

    def calculate_metrics(self, batch, mode) -> dict:
        """Override metrics calculation to include multi-vertebrae specific metrics."""
        # Get base metrics
        feature_metrics = self.feature_extraction_module.calculate_metrics(batch, mode)
        refinement_metrics = self.refinement_module.calculate_metrics(batch, mode)

        # Add multi-vertebrae specific metrics
        multi_metrics = self._calculate_multi_vertebrae_metrics(batch, mode)

        return {**feature_metrics, **refinement_metrics, **multi_metrics}

    def _calculate_multi_vertebrae_metrics(self, batch, mode):
        """Calculate vertebra-specific metrics."""
        metrics = {}

        if "n_vertebrae" not in batch or "coarse_preds" not in batch:
            return metrics

        n_pois_per_vertebra = (
            batch["n_pois_per_vertebra"] if isinstance(batch["n_pois_per_vertebra"], int) else batch["n_pois_per_vertebra"][0]
        )

        predictions = batch["coarse_preds"]
        targets = batch["target"]
        loss_mask = batch["loss_mask"]
        # Report millimetres, like every other metric in this package. These metrics
        # were unreachable until now and computed raw voxel distances, which would
        # not have been comparable with fine_mean_distance_* alongside them.
        zoom = batch["zoom"].to(targets.device).unsqueeze(1)

        # Metrics for current vertebra (first n_pois_per_vertebra POIs)
        current_preds = predictions[:, :n_pois_per_vertebra]
        current_targets = targets[:, :n_pois_per_vertebra]
        current_mask = loss_mask[:, :n_pois_per_vertebra]

        if current_mask.any():
            current_distances = torch.norm((current_preds - current_targets) * zoom, dim=-1)
            current_masked_distances = current_distances[current_mask]

            metrics[f"current_vertebra_mean_distance_{mode}"] = current_masked_distances.mean()
            metrics[f"current_vertebra_std_distance_{mode}"] = current_masked_distances.std()

        # Metrics for neighbor vertebrae (remaining POIs)
        total_pois = predictions.shape[1]
        if total_pois > n_pois_per_vertebra:
            neighbor_preds = predictions[:, n_pois_per_vertebra:]
            neighbor_targets = targets[:, n_pois_per_vertebra:]
            neighbor_mask = loss_mask[:, n_pois_per_vertebra:]

            if neighbor_mask.any():
                neighbor_distances = torch.norm((neighbor_preds - neighbor_targets) * zoom, dim=-1)
                neighbor_masked_distances = neighbor_distances[neighbor_mask]

                metrics[f"neighbor_vertebrae_mean_distance_{mode}"] = neighbor_masked_distances.mean()
                metrics[f"neighbor_vertebrae_std_distance_{mode}"] = neighbor_masked_distances.std()

                # Ratio metrics
                if current_mask.any():
                    current_mean = current_masked_distances.mean()
                    neighbor_mean = neighbor_masked_distances.mean()
                    metrics[f"neighbor_to_current_distance_ratio_{mode}"] = neighbor_mean / current_mean

        return metrics

    def predict_current_vertebra_only(self, batch) -> dict:
        """Make predictions and return only current vertebra POIs.

        Useful for inference when you only want the primary predictions.
        """
        batch = self(batch)

        if "n_pois_per_vertebra" not in batch:
            return batch

        n_pois = batch["n_pois_per_vertebra"]

        # Filter predictions to only current vertebra
        if "coarse_preds" in batch:
            batch["coarse_preds"] = batch["coarse_preds"][:, :n_pois]

        if "refined_preds" in batch:
            batch["refined_preds"] = batch["refined_preds"][:, :n_pois]

        if "target" in batch:
            batch["target"] = batch["target"][:, :n_pois]

        if "loss_mask" in batch:
            batch["loss_mask"] = batch["loss_mask"][:, :n_pois]

        return batch


#: Config ``"type"`` string -> top-level prediction module.
#:
#: ``train.py`` previously hard-coded ``PoiPredictionModule`` and ignored
#: ``module_config["type"]`` entirely, so a config asking for the neighbour-aware
#: variant silently trained the single-vertebra one. Dispatching through this
#: registry makes the config's choice actually take effect.
PREDICTION_MODULES = {
    "PoiPredictionModule": PoiPredictionModule,
    "PoiNeighborPredictionModule": PoiNeighborPredictionModule,
}
