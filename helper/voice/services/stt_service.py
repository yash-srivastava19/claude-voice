"""Speech to text: the interface every engine meets, and the registry that picks one.

An engine turns one utterance into text. It is given 16 kHz mono float32 audio
(frames.to_float) and returns the raw transcript, "" for none; cleaning and echo
checks happen downstream. transcribe may be called from several threads; engines
serialise internally if they must. Loading happens in the constructor, so a slow
model costs once, at start.

To add an engine: write services/<engine>/stt.py with a class meeting STTService,
and add a line to ENGINES. Nothing else changes.
"""

from typing import Protocol

from voice.services.registry import load


class STTService(Protocol):
    name: str

    def transcribe(self, audio) -> str: ...


ENGINES = {
    "whisper": ("voice.services.whisper.stt", "WhisperSTT"),
    "moonshine": ("voice.services.sherpa.stt", "MoonshineSTT"),
    "parakeet": ("voice.services.sherpa.stt", "ParakeetSTT"),
}


def create(name: str, **opts) -> STTService:
    return load(ENGINES, "speech-to-text", name)(**opts)
