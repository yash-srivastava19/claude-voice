"""Moonshine and Parakeet through sherpa-onnx.

model is the folder sherpa-onnx's release tarball unpacks to. Parakeet takes
hotwords: words to favour, decoded with modified beam search.
"""

import os
import threading
from pathlib import Path


class SherpaSTT:
    def __init__(self, kind: str, model: str | Path, threads: int | None = None,
                 hotwords: list[str] | None = None, hotwords_score: float = 2.0):
        import sherpa_onnx as so
        d = Path(model)
        threads = threads or min(8, os.cpu_count() or 4)
        self.name = kind
        if kind == "moonshine":
            if (d / "encoder_model.ort").exists():  # the 2026 release
                self.rec = so.OfflineRecognizer.from_moonshine_v2(
                    encoder=str(d / "encoder_model.ort"), decoder=str(d / "decoder_model_merged.ort"),
                    tokens=str(d / "tokens.txt"), num_threads=threads)
            else:
                self.rec = so.OfflineRecognizer.from_moonshine(
                    preprocessor=str(d / "preprocess.onnx"), encoder=str(d / "encode.int8.onnx"),
                    uncached_decoder=str(d / "uncached_decode.int8.onnx"),
                    cached_decoder=str(d / "cached_decode.int8.onnx"),
                    tokens=str(d / "tokens.txt"), num_threads=threads)
        elif kind == "parakeet":
            kw = {}
            if hotwords:
                path = d / "hotwords.txt"
                path.write_text("\n".join(w.upper() for w in hotwords) + "\n")
                kw = dict(decoding_method="modified_beam_search", hotwords_file=str(path),
                          hotwords_score=hotwords_score)
            self.rec = so.OfflineRecognizer.from_transducer(
                encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
                joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads,
                model_type="nemo_transducer", modeling_unit="bpe", **kw)
        else:
            raise ValueError(f"sherpa engine kind {kind!r}: moonshine or parakeet")
        self.lock = threading.Lock()

    def transcribe(self, audio) -> str:
        with self.lock:
            s = self.rec.create_stream()
            s.accept_waveform(16000, audio)
            self.rec.decode_stream(s)
        return s.result.text.strip()
