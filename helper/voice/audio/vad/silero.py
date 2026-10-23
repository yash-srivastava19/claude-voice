"""Silero VAD v5: the probability that a mic frame holds speech."""

from voice.config import MODELS
from voice.frames import RATE


class SileroVAD:
    def __init__(self, model=MODELS / "silero_vad.onnx"):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=opts)
        self.sr = np.array(RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        """Forget the stream so far, as after the mic restarts."""
        self.state = self.np.zeros((2, 1, 128), dtype=self.np.float32)
        self.context = self.np.zeros((1, 64), dtype=self.np.float32)

    def prob(self, frame: bytes) -> float:
        np = self.np
        x = (np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0).reshape(1, -1)
        out, self.state = self.session.run(None, {"input": np.concatenate([self.context, x], axis=1),
                                                  "state": self.state, "sr": self.sr})
        self.context = x[:, -64:]
        return float(out[0][0])
