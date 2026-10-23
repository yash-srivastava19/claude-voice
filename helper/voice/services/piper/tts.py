"""Piper voices through sherpa-onnx. A ref is a model folder as sherpa-onnx's release
tarball unpacks it (model .onnx, tokens.txt, espeak-ng-data/), or the name of one in
MODELS/piper/. Each voice is its own model, loaded on first use and kept."""

import os
import threading
from pathlib import Path

from voice.config import MODELS

PIPER_DIR = MODELS / "piper"


def _folder(ref) -> Path:
    path = Path(ref)
    if not path.is_absolute():
        path = PIPER_DIR / ref
    if not (path.is_dir() and any(path.glob("*.onnx")) and (path / "tokens.txt").exists()):
        raise ValueError(f"no Piper voice at {path}")
    return path


class PiperTTS:
    @staticmethod
    def check(ref):
        _folder(ref)

    def __init__(self, threads: int | None = None):
        self.threads = threads or min(4, os.cpu_count() or 4)
        self.models = {}
        self.lock = threading.Lock()

    def load_voice(self, ref):
        d = _folder(ref)
        if d not in self.models:
            import sherpa_onnx as so
            cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
                vits=so.OfflineTtsVitsModelConfig(model=str(next(d.glob("*.onnx"))), tokens=str(d / "tokens.txt"),
                                                  data_dir=str(d / "espeak-ng-data")),
                num_threads=self.threads), max_num_sentences=1)
            tts = so.OfflineTts(cfg)
            tts.generate("Sure, so here is how the whole thing fits together.", sid=0, speed=1.0)  # warm-up
            self.models[d] = tts
        return self.models[d]

    def synthesize(self, text, tts, speed):
        import numpy as np
        with self.lock:
            a = tts.generate(text, sid=0, speed=speed)
        pcm = (np.clip(np.asarray(a.samples, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
        return a.sample_rate, pcm.tobytes()
