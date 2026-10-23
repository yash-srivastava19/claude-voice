"""Lets tests import the daemon's modules without installing anything."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HELPER = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(HELPER))


def _load():
    try:  # after the split into a package
        from voice.processors import filters
        return SimpleNamespace(filters=filters)
    except ImportError:  # before it: everything is in voiced.py
        spec = importlib.util.spec_from_file_location("voiced", HELPER / "voiced.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return SimpleNamespace(filters=mod)


voice = _load()
