"""Explicit name-to-class registries for JSON-configured components.

Experiment configs address components by a ``"type"`` string::

    {"type": "SADenseNet", "params": {...}}

Resolution used to go through ``getattr(module, config["type"])``, which would
happily return *any* attribute of the module - including one that is not a model -
and whose ``AttributeError`` bypassed the intended "unknown type" message. These
registries make the set of valid names explicit and produce an error that lists
what is actually available.

The config file format is unchanged: every ``type`` string that worked before still
resolves, including the refinement ablation names, which are now aliases for flag
combinations of a single class (see :mod:`verpex.modules.refinement`).
"""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class UnknownTypeError(KeyError):
    """Raised when a config names a component that is not registered."""

    def __init__(self, kind: str, name: str, known: list[str]) -> None:
        self.kind, self.name, self.known = kind, name, known
        super().__init__(f"Unknown {kind} type {name!r}. Registered {kind} types: {', '.join(sorted(known))}.")

    def __str__(self) -> str:
        return f"Unknown {self.kind} type {self.name!r}. Registered {self.kind} types: {', '.join(sorted(self.known))}."


def resolve(registry: dict[str, T], kind: str, name: str) -> T:
    """Look ``name`` up in ``registry``.

    Args:
        registry: Mapping of config type string to entry.
        kind: Human-readable category, used in the error message.
        name: The ``"type"`` value from the config.

    Returns:
        The registered entry.

    Raises:
        UnknownTypeError: If ``name`` is not registered.
    """
    try:
        return registry[name]
    except KeyError:
        raise UnknownTypeError(kind, name, list(registry)) from None


def build(registry: dict[str, Any], kind: str, config: dict[str, Any]) -> Any:
    """Instantiate the component a config describes.

    Args:
        registry: Mapping of config type string to a class or factory.
        kind: Human-readable category, used in the error message.
        config: A ``{"type": ..., "params": {...}}`` mapping. ``params`` may be
            omitted or null.

    Returns:
        The constructed component.

    Raises:
        UnknownTypeError: If the config names an unregistered type.
        KeyError: If the config has no ``"type"`` key.
        TypeError: If ``params`` is present but is not a mapping.
    """
    if "type" not in config:
        raise KeyError(f"{kind} config is missing the required 'type' key; got keys: {', '.join(sorted(config))}.")
    factory = resolve(registry, kind, config["type"])
    # `"params": null` is easy to write by hand and would otherwise fail as an opaque
    # "argument after ** must be a mapping" TypeError.
    params = config.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError(f"{kind} config's 'params' must be a mapping, got {type(params).__name__}.")
    return factory(**params)
