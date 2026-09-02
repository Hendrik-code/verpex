"""Machine-specific path resolution.

Every filesystem location this project needs is looked up through :func:`get_path`
rather than hard-coded, so the same code runs on any machine. Values come from,
in order of precedence:

1. an environment variable, e.g. ``VERPEX_MODEL_ROOT``;
2. ``config/paths.yaml`` at the repository root (git-ignored);
3. nothing - a missing key raises :class:`PathConfigError` naming exactly what to set.

``config/paths.example.yaml`` is the committed template. Copy it to
``config/paths.yaml`` and fill in the locations for your machine.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

#: Keys a configuration file may define, with what each one is for.
KNOWN_KEYS: dict[str, str] = {
    "data_root": "Root holding the BIDS dataset(s) to read images and annotations from.",
    "cutout_root": "Where prepare_data writes per-vertebra cutouts and master_df.csv.",
    "model_root": "Root holding trained model directories and their checkpoints.",
    "output_root": "Where evaluation and inference results are written.",
    "tmp_root": "Scratch space for intermediate files.",
}

#: Only this key has a sensible machine-independent default.
_DEFAULTS: dict[str, str] = {"tmp_root": "/tmp/verpex"}

_ENV_PREFIX = "VERPEX_"


class PathConfigError(RuntimeError):
    """Raised when a required path is not configured, or is configured wrongly."""


def _repo_root() -> Path:
    """Return the repository root (the directory containing ``config/``)."""
    return Path(__file__).resolve().parents[1]


def config_file() -> Path:
    """Return the path to the user's ``paths.yaml``, whether or not it exists."""
    override = os.environ.get(f"{_ENV_PREFIX}CONFIG")
    return Path(override) if override else _repo_root() / "config" / "paths.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict[str, str]:
    """Read and validate ``paths.yaml``. Cached; call :func:`reset_cache` after edits."""
    path = config_file()
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise PathConfigError(f"{path} must contain a mapping of name -> path, got {type(loaded).__name__}.")
    unknown = set(loaded) - set(KNOWN_KEYS)
    if unknown:
        raise PathConfigError(f"{path} defines unknown key(s): {', '.join(sorted(unknown))}. Known keys: {', '.join(sorted(KNOWN_KEYS))}.")
    return {k: str(v) for k, v in loaded.items() if v is not None}


def reset_cache() -> None:
    """Forget the cached config, so a later :func:`get_path` re-reads the file."""
    _load_config.cache_clear()


def env_var_for(key: str) -> str:
    """Return the environment variable name that overrides ``key``."""
    return f"{_ENV_PREFIX}{key.upper()}"


def get_path(key: str, *, must_exist: bool = False) -> Path:
    """Resolve a configured path.

    Args:
        key: One of :data:`KNOWN_KEYS`, e.g. ``"model_root"``.
        must_exist: If true, raise unless the resolved path exists on disk.

    Returns:
        The resolved path.

    Raises:
        PathConfigError: If ``key`` is unknown, unconfigured, or ``must_exist`` is
            set and the path is absent. The message names the environment variable
            and the config file, so the fix is obvious at the point of failure
            rather than as a FileNotFoundError deep inside preprocessing.
    """
    if key not in KNOWN_KEYS:
        raise PathConfigError(f"Unknown path key {key!r}. Known keys: {', '.join(sorted(KNOWN_KEYS))}.")

    value = os.environ.get(env_var_for(key)) or _load_config().get(key) or _DEFAULTS.get(key)
    if not value:
        raise PathConfigError(
            f"Path {key!r} is not configured ({KNOWN_KEYS[key]})\n"
            f"  Set the environment variable {env_var_for(key)}, or add\n"
            f"      {key}: /your/path\n"
            f"  to {config_file()}\n"
            f"  See config/paths.example.yaml for a template."
        )

    path = Path(value).expanduser()
    if must_exist and not path.exists():
        raise PathConfigError(f"Path {key!r} resolves to {path}, which does not exist. Set {env_var_for(key)} or edit {config_file()}.")
    return path
