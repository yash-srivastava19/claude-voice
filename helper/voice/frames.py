"""The data every part passes around.

Audio is bytes of 16-bit little-endian mono PCM. Mic audio is always RATE Hz, read in
FRAME-sample frames; speech to text takes it as float32 via to_float. Text to speech
returns (rate, pcm) at whatever rate its model uses. Events for the harness are plain
dicts made by event().
"""

import time

RATE = 16000
FRAME = 512  # Silero v5 frame at 16 kHz, 32 ms
FRAME_S = FRAME / RATE


# What a mic stream yields between frames when its device drops out and comes back.
MIC_LOST, MIC_BACK = "lost", "back"


def seconds(pcm: bytes, rate: int = RATE) -> float:
    return len(pcm) / 2 / rate


def to_float(pcm: bytes):
    """16-bit PCM bytes as a float32 array in [-1, 1)."""
    import numpy as np
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def event(kind: str, **data) -> dict:
    return {"kind": kind, "at": time.time(), **data}
