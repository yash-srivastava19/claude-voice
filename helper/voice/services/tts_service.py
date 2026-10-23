"""Text to speech: engines, and the speaker that picks one per voice.

An engine (TTSEngine) owns a model and nothing else. check(ref) validates a ref
without loading the model (for --check-voice); load_voice(ref) turns a ref into
whatever the engine needs; synthesize returns (rate, pcm): 16-bit mono PCM bytes at
the engine's own sample rate, or (0, b"") for nothing to play. Each raises ValueError
for a ref it cannot use. An engine loads and warms its model in the constructor.

Speaker is what the pipeline holds (TTSService). It keeps the current voice and
speed, loads an engine the first time a voice needs it, and keeps it loaded, so
switching between voices of loaded engines is instant.

To add an engine: write services/<engine>/tts.py with a class meeting TTSEngine, and
add a line to ENGINES. Voices for it then go in services/voices.py.
"""

import importlib
from typing import Any, Protocol

from voice.log import log
from voice.services.voices import VoiceSpec, resolve


class TTSEngine(Protocol):
    def load_voice(self, ref: str) -> Any: ...

    def synthesize(self, text: str, voice: Any, speed: float) -> tuple[int, bytes]: ...


class TTSService(Protocol):
    voice: VoiceSpec
    speed: float

    def configure(self, voice: str | None = None, speed: float | None = None) -> None: ...

    def synthesize(self, text: str) -> tuple[int, bytes]: ...


# engine name -> (module, class). Imported on first use, so an engine's packages are
# only needed if one of its voices is.
ENGINES = {
    "kokoro": ("voice.services.kokoro.tts", "KokoroTTS"),
    "piper": ("voice.services.piper.tts", "PiperTTS"),
    "command": ("voice.services.command.tts", "CommandTTS"),
}


def _engine_class(name):
    try:
        module, cls = ENGINES[name]
    except KeyError:
        raise ValueError(f"no text-to-speech engine {name!r}; one of {', '.join(ENGINES)}") from None
    return getattr(importlib.import_module(module), cls)


def check(name: str) -> VoiceSpec:
    """The voice a name means, if its engine can use it; raises ValueError otherwise."""
    spec = resolve(name)
    _engine_class(spec.engine).check(spec.ref)
    return spec


class Speaker:
    def __init__(self, voice: str, speed: float, fallback: str = "michael"):
        self.engines: dict[str, TTSEngine] = {}
        self.speed = speed
        try:
            self._use(voice)
        except ValueError as err:
            log(f"voice {voice!r}: {err}; using {fallback}")
            self._use(fallback)

    @property
    def name(self):
        return self.voice.engine

    def _use(self, name):
        spec = check(name)
        engine = self.engines.get(spec.engine)
        if engine is None:
            engine = self.engines[spec.engine] = _engine_class(spec.engine)()
        self.loaded = engine.load_voice(spec.ref)
        self.engine, self.voice = engine, spec

    def configure(self, voice=None, speed=None):
        """Applies a new voice and/or speed; raises ValueError if either is bad."""
        if speed is not None:
            speed = float(speed)
            if not 0.5 <= speed <= 2.0:
                raise ValueError("speed must be between 0.5 and 2.0")
        if voice is not None and voice != self.voice.name:
            self._use(voice)
            log(f"voice: {self.voice.name}")
        if speed is not None:
            self.speed = speed

    def synthesize(self, text):
        return self.engine.synthesize(text, self.loaded, self.speed)
