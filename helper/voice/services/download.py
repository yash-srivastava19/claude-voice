"""Fetches a model folder packed as a .tar.bz2 (sherpa-onnx's release layout) into MODELS."""

import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from voice.config import MODELS


def fetch_folder(url: str, dest: Path):
    """Downloads and unpacks url so that dest holds the tarball's one top folder. Safe to rerun."""
    if dest.is_dir():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=MODELS) as tmp:
        archive = Path(tmp) / "voice.tar.bz2"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        (top,) = [p for p in Path(tmp).iterdir() if p.is_dir()]
        shutil.move(str(top), dest)
