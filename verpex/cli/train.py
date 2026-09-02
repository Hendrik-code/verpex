"""Train a POI prediction model from a JSON experiment config."""

import argparse
import json
import os

# cuBLAS reads this when it initialises its CUDA context, which happens on the first
# CUDA op. Setting it after that point has no effect and
# torch.use_deterministic_algorithms(True) then raises on the first matmul, so it has
# to be set before torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytorch_lightning as pl
import torch

from verpex.modules.data_modules import create_data_module
from verpex.modules.poi_module import PREDICTION_MODULES
from verpex.registry import build
from verpex.training_utils import create_callbacks, save_data_module_config

#: Seed used when neither the config nor the command line sets one.
DEFAULT_SEED = 42


def run_experiment(experiment_config, seed: int | None = None) -> None:
    """Train one model from a JSON experiment config.

    Seeds every RNG and enables deterministic algorithms, so two runs of the same
    config and seed produce the same weights.

    Args:
        experiment_config: A parsed experiment config, with ``data_module_config``,
            ``module_config``, ``callbacks_config`` and ``trainer_config`` keys.
            An optional ``seed`` key sets the seed.
        seed: Overrides the config's ``seed``. Defaults to 42 when neither is given.
    """
    if seed is None:
        seed = experiment_config.get("seed", DEFAULT_SEED)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")
    # warn_only=False raises on the first non-deterministic op rather than letting 3D
    # CUDA convolutions silently introduce run-to-run variance.
    torch.use_deterministic_algorithms(mode=True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_module = create_data_module(experiment_config["data_module_config"])
    data_module.setup()

    poi_module = build(PREDICTION_MODULES, "prediction module", experiment_config["module_config"])

    logger = pl.loggers.TensorBoardLogger(save_dir=experiment_config["path"], name=experiment_config["name"])
    save_data_module_config(data_module, logger.log_dir)

    callbacks = create_callbacks(experiment_config.get("callbacks_config", []))
    trainer_config = experiment_config.get("trainer_config", {})
    trainer_config.setdefault("callbacks", callbacks)
    trainer_config.setdefault("deterministic", "warn")
    trainer_config["logger"] = logger

    trainer = pl.Trainer(**trainer_config)

    print("\n=== Starting Training ===")
    trainer.fit(
        poi_module,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    print("\n=== Final Debug Info ===")
    print(f"Epochs completed: {trainer.current_epoch}")
    print(f"Max epochs allowed: {trainer.max_epochs}")


def main() -> None:
    """Train a POI prediction model from a JSON experiment config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Experiment config file")
    parser.add_argument("--config-dir", type=str, help="Directory containing experiment config files")
    parser.add_argument("--seed", type=int, default=None, help=f"Random seed; overrides the config's 'seed' (default {DEFAULT_SEED})")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            run_experiment(json.load(f), seed=args.seed)

    if args.config_dir:
        for config_file in sorted(os.listdir(args.config_dir)):
            with open(os.path.join(args.config_dir, config_file)) as f:
                run_experiment(json.load(f), seed=args.seed)


if __name__ == "__main__":
    main()
