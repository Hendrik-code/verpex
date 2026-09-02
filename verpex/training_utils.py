"""Helpers for assembling a training run from a JSON experiment config."""

from __future__ import annotations

import json
import os
from typing import Any

import pytorch_lightning as pl

from verpex.registry import build

#: Config ``"type"`` string -> Lightning callback.
#:
#: Previously ``create_callbacks`` matched two names with no ``else`` branch, so a
#: typo in ``callbacks_config`` silently produced a run with no checkpointing and
#: no early stopping. Going through the registry turns that into a clear error.
CALLBACKS = {
    "ModelCheckpoint": pl.callbacks.ModelCheckpoint,
    "EarlyStopping": pl.callbacks.EarlyStopping,
    "LearningRateMonitor": pl.callbacks.LearningRateMonitor,
}


def save_data_module_config(data_module: pl.LightningDataModule, save_path: str | os.PathLike) -> None:
    """Write a data module's hyperparameters next to the training logs.

    Args:
        data_module: The configured data module.
        save_path: Directory to write ``data_module_params.json`` into. Created if absent.
    """
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "data_module_params.json"), "w", encoding="utf-8") as handle:
        json.dump(data_module.hparams, handle, indent=4)


def create_callbacks(callbacks_config: list[dict[str, Any]]) -> list[pl.Callback]:
    """Build the Lightning callbacks a config lists.

    Args:
        callbacks_config: A list of ``{"type": ..., "params": {...}}`` mappings.

    Returns:
        The constructed callbacks, in config order.

    Raises:
        UnknownTypeError: If a config names a callback that is not registered.
    """
    return [build(CALLBACKS, "callback", config) for config in callbacks_config]
