"""Keeps text to speech out of speech to text's way. When both run (you talk over a
reply), Kokoro at equal priority slows your transcription 0.50 s -> 0.93 s; at lower
priority it stays 0.60 s (bench/contention.py). The reply is paused meanwhile, so the
slower synthesis is not heard.

Linux sets priority per thread, so this lowers only the threads text to speech uses:
the one that feeds it, and the worker pool a model starts when it loads. Elsewhere it
does nothing.
"""

import os
import threading
from contextlib import contextmanager

from voice.config import TTS_NICE


def _threads():
    try:
        return set(os.listdir("/proc/self/task"))
    except OSError:
        return set()


def _lower(tid):
    try:
        os.setpriority(os.PRIO_PROCESS, int(tid), TTS_NICE)
    except (OSError, AttributeError):
        pass


def lower_this_thread():
    if TTS_NICE:
        _lower(threading.get_native_id())


@contextmanager
def lower_new_threads():
    """Threads started inside the block, such as a model's worker pool, run at TTS_NICE."""
    before = _threads()
    yield
    if TTS_NICE:
        for tid in _threads() - before:
            _lower(tid)
