"""whisper.cpp through pywhispercpp."""

import os
import threading

from voice.frames import RATE


class WhisperSTT:
    name = "whisper"

    def __init__(self, model: str, threads: int | None = None, short_window: bool = False, **options):
        """options go to every transcribe call (initial_prompt, single_segment, max_tokens, ...).

        short_window sizes whisper's audio window to the clip instead of its fixed 30 s,
        which is most of the cost of a short utterance.
        """
        from pywhispercpp.model import Model
        self.model = Model(model, n_threads=threads or min(8, os.cpu_count() or 4), print_progress=False,
                           print_realtime=False, redirect_whispercpp_logs_to=None)
        self.short_window = short_window
        self.options = options
        self.lock = threading.Lock()  # one whisper context: one transcription at a time

    def transcribe(self, audio) -> str:
        opts = dict(self.options)
        if self.short_window:
            opts["audio_ctx"] = min(1500, int(len(audio) / RATE * 50) + 64)
        with self.lock:
            segs = self.model.transcribe(audio, **opts)
        return " ".join(seg.text.strip() for seg in segs)
