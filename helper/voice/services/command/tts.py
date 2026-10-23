"""Your own text to speech: an executable that reads text on stdin and writes a WAV.
It gets the speaking rate in VOICE_SPEED."""

import io
import os
import subprocess
import wave


class CommandTTS:
    @staticmethod
    def check(ref):
        if not os.access(ref, os.X_OK):
            raise ValueError(f"{ref} is not an executable file")

    def load_voice(self, ref):
        self.check(ref)
        return ref

    def synthesize(self, text, command, speed):
        import numpy as np
        res = subprocess.run([command], input=text.encode(), capture_output=True, timeout=60,
                             env={**os.environ, "VOICE_SPEED": str(speed)})
        if res.returncode != 0:
            raise RuntimeError(f"{command} exited {res.returncode}: {res.stderr.decode()[-200:]}")
        with wave.open(io.BytesIO(res.stdout)) as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if width != 2:
            raise RuntimeError(f"{command}: write 16-bit WAV (got {8 * width}-bit)")
        pcm = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return rate, pcm.tobytes()
