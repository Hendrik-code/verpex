# Contributing to verpex

Thanks for helping out. This document covers how to set up a working checkout, what
the automated checks expect, and the handful of conventions in this codebase that
are easy to break by accident.

## Getting set up

Python 3.10, 3.11 or 3.12 (see `pyproject.toml`). The package must be installed —
several tools, `mypy` included, follow the real imports.

```bash
conda create -n verpex python=3.11
conda activate verpex

pip install -e .
pip install pytest ruff mypy pre-commit pandas-stubs types-PyYAML

pre-commit install
```

`pandas-stubs` and `types-PyYAML` are type-only packages with no runtime effect;
without them mypy reports missing stubs for `pandas` and `yaml`.

Machine-specific paths go in `config/paths.yaml`, which is git-ignored:

```bash
cp config/paths.example.yaml config/paths.yaml
```

You do not need this to run the test suite — see [Tests](#tests) — but you do need
it for anything that touches real data.

## Before you commit

`pre-commit install` runs these automatically. To run them by hand:

```bash
ruff check . && ruff format --check .
mypy
pytest
```

All three must be clean. There are currently **zero** ruff findings, **zero** mypy
errors and **zero** failing or skipped tests, so any output at all is something you
introduced.

A note on `ruff`: the version is pinned in **both** `.pre-commit-config.yaml` and
`.github/workflows/lint.yml`. Different ruff versions enforce different rulesets, so
if you bump one, bump the other in the same commit — otherwise CI and your local
hooks will disagree about whether your code passes. The same applies to `mypy`,
which runs as a `language: system` hook (in your project environment, because
`torch` is far too heavy to reinstall into an isolated hook venv).

## Tests

Tests live in `unit_tests/`. They run on synthetic tensors and need **no dataset, no
checkpoint and no GPU** — this is deliberate, and new tests should keep it that way.
The shared fixtures in `unit_tests/conftest.py` build deliberately small models
(`MODEL_DIMS`); note the comment there about `patch_size` needing to stay ≥ 16.

Conventions worth matching:

- **Name the behaviour, not the function.** Existing tests read like
  `test_refined_predictions_keep_sub_voxel_precision` and
  `test_loss_scales_with_voxel_spacing`, not `test_forward_2`.
- **Prefer exact expected values to tolerances.** `unit_tests/test_metrics.py` picks
  fixture errors of `[1, 2, ..., 8]` precisely so every aggregate has an exact
  answer.
- **When you fix a bug, add the test that would have caught it**, and check that it
  actually fails against the old code before you keep it. Several tests carry a
  docstring naming the regression they guard.

Coverage is available (`coverage` is in the dev group) but no CI job enforces a
threshold.

## Adding a component

Models, refinement modules, data modules and callbacks are addressed from JSON
experiment configs by a `"type"` string, resolved through an explicit registry
rather than `getattr` — see the module docstring in `verpex/registry.py` for why.

To add one:

1. Write the class.
2. Register it in the relevant dict: `PREDICTION_MODULES`
   (`verpex/modules/poi_module.py`), `FEATURE_EXTRACTION_MODULES`
   (`verpex/modules/feature_extraction.py`), `REFINEMENT_MODULES`
   (`verpex/modules/refinement.py`), `DATA_MODULES`
   (`verpex/modules/data_modules.py`) or `CALLBACKS`
   (`verpex/training_utils.py`).
3. Add its name to `HISTORICAL_TYPES` in `unit_tests/test_registry.py`.

Renaming a registered type string breaks every experiment config that uses it, so
treat the names as a public interface: rename only deliberately, and update
`HISTORICAL_TYPES` in the same commit so the guard reflects the new set.

## Conventions that are easy to break

- **Errors and distances are in millimetres, everywhere.** Model predictions are in
  voxels, so anything reported as a metric or a loss must be scaled by the batch's
  `zoom` (mm per voxel) first. A module that forgets this produces numbers that look
  plausible and are silently incomparable with every other run. `test_metrics.py`
  and `test_refinement.py` both assert in mm.
- **`master_df` paths are relative to the cutout root.** `prepare_data` writes
  `file_dir` relative so the CSV survives the data moving;
  `verpex.data.dataset.resolve_cutout_dir` resolves it against the configured
  `cutout_root`, passing absolute entries (from older versions) through unchanged.
  Do not write absolute paths into a new `master_df`.
- **`verpex/data/transforms.py` has a vendored region.** Below the marked divider is
  a copy of MONAI's affine-transform internals, kept close to upstream so it stays
  easy to diff. It is exempted from both ruff and mypy in `pyproject.toml`. Put your
  own transforms above the divider, and keep edits below it minimal and marked.
- **`spconv` is optional.** The sparse-convolution backbones (`SMDenseNet`,
  `SMSADenseNet`) import it lazily inside `__init__` so the dense pipeline never
  touches it. `verpex/models/subm_densenet.py` therefore fails to import without
  `spconv` installed, which is expected — do not "fix" it by moving the import, and
  do not add a top-level `spconv` import anywhere else.
- **Data-module setup runs on defaults.** `transform_config` and several other
  options default to `None`; make sure a new code path still works when they are
  unset.

## Pull requests

- Branch off `main`; both CI workflows (`lint` and `tests`) run on pushes to `main`
  and on pull requests against it.
- The `tests` workflow runs the suite on Python 3.10, 3.11 and 3.12. Code that
  relies on a 3.11+ feature will fail the matrix.
- Keep the diff to the change you are making. If you spot an unrelated problem,
  mention it rather than folding it in.
- Explain *why* in the commit message and the PR description. This codebase carries
  a lot of comments that record why something is the way it is; that context has
  been more valuable than the code itself more than once.

## Releasing

The version is not written in `pyproject.toml` — it is derived from the latest git
tag at build time by
[poetry-dynamic-versioning](https://github.com/mtkennerly/poetry-dynamic-versioning).
`version = "0.0.0"` is a placeholder.

```bash
git tag v0.1.0 && git push --tags
```

Between tags the version reads as `0.1.0.post<n>.dev0+<sha>`. A repository with **no
tags at all** builds as `0.0.0.post<n>.dev0+<sha>`, so tag once after the initial
commit.
