"""Piper voices through sherpa-onnx. A ref is a model folder as sherpa-onnx's release
tarball unpacks it (model .onnx, tokens.txt, espeak-ng-data/), or the name of one in
MODELS/piper/. Each voice is its own model, loaded on first use and kept."""

import threading
from pathlib import Path

from voice.config import MODELS

PIPER_DIR = MODELS / "piper"


class NotDownloaded(ValueError):
    pass


def folder(ref) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else PIPER_DIR / ref


def _folder(ref) -> Path:
    path = folder(ref)
    if not path.exists():
        raise NotDownloaded(f"not downloaded yet ({path})")
    if not (any(path.glob("*.onnx")) and (path / "tokens.txt").exists()):
        raise ValueError(f"no Piper voice at {path}")
    return path


class PiperTTS:
    @staticmethod
    def check(ref):
        import importlib.util
        if importlib.util.find_spec("sherpa_onnx") is None:  # before the folder, so nothing downloads in vain
            raise ValueError("Piper voices need the updated install, which downloads in the background "
                             "after /talk start; try again in a few minutes")
        _folder(ref)

    def __init__(self, threads: int | None = None, provider: str = "cpu"):
        self.threads = threads or 2  # leaves the cores to speech to text; still 4x faster than real time
        self.provider = provider
        self.models = {}
        self.lock = threading.Lock()

    def load_voice(self, ref):
        d = _folder(ref)
        if d not in self.models:
            import sherpa_onnx as so
            cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
                vits=so.OfflineTtsVitsModelConfig(model=str(next(d.glob("*.onnx"))), tokens=str(d / "tokens.txt"),
                                                  data_dir=str(d / "espeak-ng-data")),
                num_threads=self.threads, provider=self.provider), max_num_sentences=1)
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
