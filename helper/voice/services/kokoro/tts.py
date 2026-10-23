"""Kokoro through kokoro-onnx."""

from voice.config import KOKORO_MODEL, KOKORO_VOICES
from voice.services.kokoro.voices import style_for


class KokoroTTS:
    @staticmethod
    def check(ref):
        import numpy as np
        style_for(ref, np)

    def __init__(self, model: str = KOKORO_MODEL, provider: str = "cpu", threads: int | None = None):
        """provider: "cpu", "cuda" or "coreml" (needs onnxruntime built with it)."""
        import numpy as np
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        self.np = np
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
        providers = {"cuda": ["CUDAExecutionProvider"], "coreml": ["CoreMLExecutionProvider"]}.get(provider, [])
        session = ort.InferenceSession(model, sess_options=opts, providers=providers + ["CPUExecutionProvider"])
        self.model = Kokoro.from_session(session, str(KOKORO_VOICES))
        self.styles = np.load(KOKORO_VOICES)
        # Warm up on a full sentence: the first real call is otherwise about twice as slow.
        for _ in range(2):
            self.model.create("Sure, so here is how the whole thing fits together.",
                              voice=self.styles["am_michael"], lang="en-us")

    def load_voice(self, ref):
        return style_for(ref, self.np, self.styles)

    def synthesize(self, text, voice, speed):
        style, lang = voice
        audio, rate = self.model.create(text, voice=style, speed=speed, lang=lang)
        return rate, (self.np.clip(audio, -1, 1) * 32767).astype(self.np.int16).tobytes()
