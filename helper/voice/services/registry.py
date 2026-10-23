"""Engines by name, imported on first use: an engine's packages are only needed if it is."""

import importlib


def load(engines: dict, kind: str, name: str):
    """The class registered under name in engines {name: (module, class)}."""
    try:
        module, cls = engines[name]
    except KeyError:
        raise ValueError(f"no {kind} engine {name!r}; one of {', '.join(engines)}") from None
    return getattr(importlib.import_module(module), cls)
