"""Path configuration: precedence, validation and error quality."""

from __future__ import annotations

import pytest

from verpex import paths
from verpex.paths import KNOWN_KEYS, PathConfigError, env_var_for, get_path


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Point the loader at an empty temp config and clear all env overrides."""
    config = tmp_path / "paths.yaml"
    monkeypatch.setenv("VERPEX_CONFIG", str(config))
    for key in KNOWN_KEYS:
        monkeypatch.delenv(env_var_for(key), raising=False)
    paths.reset_cache()
    yield config
    paths.reset_cache()


def write(config, text):
    config.write_text(text, encoding="utf-8")
    paths.reset_cache()


def test_env_var_wins_over_the_config_file(_isolated_config, monkeypatch):
    write(_isolated_config, "model_root: /from/yaml\n")
    monkeypatch.setenv("VERPEX_MODEL_ROOT", "/from/env")
    assert str(get_path("model_root")) == "/from/env"


def test_config_file_is_used_when_no_env_var(_isolated_config):
    write(_isolated_config, "model_root: /from/yaml\n")
    assert str(get_path("model_root")) == "/from/yaml"


def test_tmp_root_has_a_default():
    assert get_path("tmp_root")


def test_missing_key_names_the_env_var_and_the_file(_isolated_config):
    write(_isolated_config, "{}\n")
    with pytest.raises(PathConfigError) as excinfo:
        get_path("model_root")
    message = str(excinfo.value)
    assert "VERPEX_MODEL_ROOT" in message
    assert "model_root" in message
    assert "paths.example.yaml" in message


def test_unknown_key_is_rejected():
    with pytest.raises(PathConfigError, match="Unknown path key"):
        get_path("not_a_real_key")


def test_unknown_key_in_config_file_is_rejected(_isolated_config):
    write(_isolated_config, "typo_root: /somewhere\n")
    with pytest.raises(PathConfigError, match="unknown key"):
        get_path("model_root")


def test_must_exist_reports_the_resolved_path(_isolated_config):
    write(_isolated_config, "model_root: /definitely/not/here\n")
    with pytest.raises(PathConfigError, match="does not exist"):
        get_path("model_root", must_exist=True)


def test_user_home_is_expanded(_isolated_config):
    write(_isolated_config, "model_root: ~/models\n")
    assert "~" not in str(get_path("model_root"))


def test_repo_root_is_the_directory_holding_pyproject(monkeypatch):
    """`_repo_root` must not walk past the repository.

    It previously used ``parents[2]``, correct only for the old ``src/verpex/``
    layout; under the flat layout that resolved to the repo's *parent*, so the
    documented ``config/paths.yaml`` was silently never found. The autouse fixture
    above always sets ``VERPEX_CONFIG``, which is why nothing else catches this.
    """
    monkeypatch.delenv("VERPEX_CONFIG", raising=False)
    paths.reset_cache()

    root = paths._repo_root()
    assert (root / "pyproject.toml").is_file(), f"{root} is not the repository root"
    assert paths.config_file() == root / "config" / "paths.yaml"
