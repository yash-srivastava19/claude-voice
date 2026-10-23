"""Speech to text: the interface every engine meets, and the registry that picks one.

An engine turns one utterance into text. It is given 16 kHz mono float32 audio
(frames.to_float) and returns the raw transcript, "" for none; cleaning and echo
checks happen downstream. transcribe may be called from several threads; engines
serialise internally if they must. Loading happens in the constructor, so a slow
model costs once, at start.

To add an engine: write services/<engine>/stt.py with a class meeting STTService,
and add a line to ENGINES. Nothing else changes.
"""

from typing import Callable, Protocol


class STTService(Protocol):
    name: str

    def transcribe(self, audio) -> str: ...


def _whisper(**opts) -> STTService:
    from voice.services.whisper.stt import WhisperSTT
    return WhisperSTT(**opts)


def _sherpa(kind):
    def make(**opts) -> STTService:
        from voice.services.sherpa.stt import SherpaSTT
        return SherpaSTT(kind, **opts)
    return make


ENGINES: dict[str, Callable[..., STTService]] = {
    "whisper": _whisper,
    "moonshine": _sherpa("moonshine"),
    "parakeet": _sherpa("parakeet"),
}


def create(name: str, **opts) -> STTService:
    try:
        factory = ENGINES[name]
    except KeyError:
        raise ValueError(f"no speech-to-text engine {name!r}; one of {', '.join(ENGINES)}") from None
    return factory(**opts)
