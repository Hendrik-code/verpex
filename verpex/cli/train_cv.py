"""Train a POI prediction model with k-fold cross-validation."""

import argparse
import json
import os

# Must be set before torch is imported; see the note in verpex.cli.train.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold

from verpex.cli.train import DEFAULT_SEED
from verpex.modules.data_modules import create_data_module
from verpex.modules.poi_module import PREDICTION_MODULES
from verpex.registry import build
from verpex.training_utils import create_callbacks, save_data_module_config


def run_cv(n_folds, experiment_config, save_predictions=False, poi_file_ending=None, seed: int | None = None) -> None:
    """Train one model per cross-validation fold.

    The configured train and validation subject lists are merged and re-split into
    ``n_folds`` random folds; each fold trains into its own ``fold_{i}`` directory.

    Args:
        n_folds: Number of cross-validation folds.
        experiment_config: A parsed experiment config. Not modified.
        save_predictions: Export each fold's predictions as pseudo-labels.
        poi_file_ending: File ending for those predictions. Required with
            ``save_predictions``.
        seed: Seeds the RNGs and the fold split. Overrides the config's ``seed``.

    Raises:
        ValueError: If ``save_predictions`` is set without ``poi_file_ending``.
    """
    # If the predictions are saved the file ending must be set
    if save_predictions and not poi_file_ending:
        raise ValueError("If predictions are saved the poi file ending must be set")

    if seed is None:
        seed = experiment_config.get("seed", DEFAULT_SEED)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")
    # Matches verpex.cli.train: raise rather than silently run non-deterministically.
    torch.use_deterministic_algorithms(mode=True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Concatenate into a new list: `+=` would append the validation subjects to the
    # list held inside experiment_config, corrupting it for any later use.
    train_subjects = list(experiment_config["data_module_config"]["params"]["train_subjects"])
    train_subjects += experiment_config["data_module_config"]["params"]["val_subjects"]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_subjects)):
        train_subjects_fold = [train_subjects[i] for i in train_idx]
        val_subjects_fold = [train_subjects[i] for i in val_idx]

        data_module_config = experiment_config["data_module_config"]
        data_module_config["params"]["train_subjects"] = train_subjects_fold
        data_module_config["params"]["val_subjects"] = val_subjects_fold

        data_module = create_data_module(data_module_config)
        data_module.setup()
        poi_module = build(PREDICTION_MODULES, "prediction module", experiment_config["module_config"])

        # Create callbacks from configuration
        callbacks = create_callbacks(experiment_config.get("callbacks_config", []))

        # Trainer configuration
        trainer_config = experiment_config.get("trainer_config", {})
        trainer_config["callbacks"] = callbacks
        trainer_config.setdefault("deterministic", "warn")

        # Add fold to path
        path = experiment_config["path"] + f"/fold_{fold}"
        trainer_config["logger"] = pl.loggers.TensorBoardLogger(path, name=experiment_config["name"])

        trainer = pl.Trainer(**trainer_config)

        # Save DataModule config. Lightning leaves `logger` unset when logging is
        # disabled, in which case there is no run directory to write beside.
        if trainer.logger is not None and trainer.logger.log_dir is not None:
            save_data_module_config(data_module, trainer.logger.log_dir)

        trainer.fit(
            poi_module,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )

        if save_predictions:
            # `create_self_training_pois` was removed from eval.py in commit 4632148
            # ("added zoom consideration to model and eval.py") without updating this
            # call site, which left train_cv.py raising ImportError on every run. The
            # function predated that zoom-correctness pass, so restoring it means
            # re-deriving it against the corrected zoom handling, not reverting 4632148.
            raise NotImplementedError(
                "--save-predictions (self-training pseudo-label export) is not available: "
                "its implementation was removed during the zoom-handling fix and has not "
                "been re-derived against the corrected zoom convention. Run the "
                "cross-validation without this flag."
            )


def main() -> None:
    """Train a POI prediction model with k-fold cross-validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, help="Number of folds for cross-validation")
    parser.add_argument("--save-predictions", action="store_true", help="Save predictions for each fold")
    parser.add_argument(
        "--poi-file-ending",
        type=str,
        help="Ending of the poi file to save predictions to",
    )
    parser.add_argument("--config", type=str, help="Experiment config file")
    parser.add_argument("--config-dir", type=str, help="Directory containing experiment config files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed; overrides the config's 'seed'")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            experiment_config = json.load(f)
            run_cv(
                args.n_folds,
                experiment_config,
                save_predictions=args.save_predictions,
                poi_file_ending=args.poi_file_ending,
                seed=args.seed,
            )

    if args.config_dir:
        for config_file in sorted(os.listdir(args.config_dir)):
            with open(os.path.join(args.config_dir, config_file)) as f:
                experiment_config = json.load(f)
                run_cv(
                    args.n_folds,
                    experiment_config,
                    save_predictions=args.save_predictions,
                    poi_file_ending=args.poi_file_ending,
                    seed=args.seed,
                )


if __name__ == "__main__":
    main()
