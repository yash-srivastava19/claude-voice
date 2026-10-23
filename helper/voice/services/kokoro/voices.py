"""Kokoro voice refs: a built-in id ("af_sky"), a mix ("af_heart:0.6+bf_emma:0.4"),
or a path to a style file (.npy, .pt)."""

from pathlib import Path

from voice.config import KOKORO_VOICES

# Kokoro's prefix: language, then f/m. Only English is wired to the transcriber.
LANGS = {"a": "en-us", "b": "en-gb"}


def kokoro_ids():
    import numpy as np
    with np.load(KOKORO_VOICES) as z:
        return sorted(k for k in z.files if k[:1] in LANGS)


def style_for(ref, np, styles=None):
    """(style vector, language) for a ref; raises ValueError naming what is wrong."""
    path = Path(ref)
    if path.suffix in (".npy", ".pt") and path.exists():
        style = np.load(path) if path.suffix == ".npy" else load_pt(path)
        return np.asarray(style, dtype=np.float32).reshape(-1, 1, 256), "en-us"
    if styles is None:
        styles = np.load(KOKORO_VOICES)
    parts, total, mix = [], 0.0, None
    for part in ref.split("+"):
        vid, _, share = part.strip().partition(":")
        if vid not in styles.files or vid[:1] not in LANGS:
            raise ValueError(f"no voice named {vid!r}; /talk voices lists them")
        try:
            weight = float(share) if share else 1.0
        except ValueError:
            raise ValueError(f"bad share {share!r} in {part!r}; write it like af_heart:0.6") from None
        parts.append(vid)
        total += weight
        mix = styles[vid] * weight if mix is None else mix + styles[vid] * weight
    if total <= 0:
        raise ValueError("the shares in a mix must add up to more than 0")
    return (mix / total).astype(np.float32), LANGS[parts[0][0]]


def load_pt(path):
    try:
        import torch
    except ImportError:
        raise ValueError(f"{path.name} is a PyTorch file; convert it once to .npy (see the README)") from None
    return torch.load(path, map_location="cpu").numpy()
