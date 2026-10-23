"""Kokoro through kokoro-onnx."""

from voice.config import KOKORO_MODEL, KOKORO_VOICES
from voice.services.kokoro.voices import style_for


class KokoroTTS:
    @staticmethod
    def check(ref):
        import numpy as np
        style_for(ref, np)

    def __init__(self, model: str = KOKORO_MODEL):
        import numpy as np
        from kokoro_onnx import Kokoro
        self.np = np
        self.model = Kokoro(model, str(KOKORO_VOICES))
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
