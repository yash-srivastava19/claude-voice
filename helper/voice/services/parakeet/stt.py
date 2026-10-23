"""NVIDIA's Parakeet TDT 0.6B v2 (int8) through sherpa-onnx.

model is the folder sherpa-onnx's release tarball unpacks to. provider is "cpu", "cuda"
or "coreml"; the last two need a sherpa-onnx build with that support.
"""

import os
import threading
from pathlib import Path

from voice.frames import RATE


class ParakeetSTT:
    name = "parakeet"

    def __init__(self, model: str | Path, threads: int | None = None, provider: str = "cpu"):
        import sherpa_onnx as so
        d = Path(model)
        # 6 beat 8 on a 2P+8E laptop, alone (0.95 s vs 1.31 s) and with TTS running (bench/run.py contention).
        threads = threads or min(6, os.cpu_count() or 4)
        self.rec = so.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads,
            model_type="nemo_transducer", modeling_unit="bpe", provider=provider)
        self.lock = threading.Lock()  # one recognizer: one decode at a time

    def transcribe(self, audio) -> str:
        with self.lock:
            stream = self.rec.create_stream()
            stream.accept_waveform(RATE, audio)
            self.rec.decode_stream(stream)
        return stream.result.text.strip()
