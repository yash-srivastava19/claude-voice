"""Text to speech: engines, and the switcher that picks one per voice.

An engine (TTSEngine) owns a model and nothing else. check(ref) validates a ref
without loading the model (for --check-voice); load_voice(ref) turns a ref into
whatever the engine needs; synthesize returns (rate, pcm): 16-bit mono PCM bytes at
the engine's own sample rate, or (0, b"") for nothing to play. Each raises ValueError
for a ref it cannot use. An engine loads and warms its model in the constructor.

TTSSwitcher is what the pipeline holds (TTSService). It keeps the current voice and
speed and loads the engine the voice needs, so
switching between voices of the same engine is instant.

To add an engine: write services/<engine>/tts.py with a class meeting TTSEngine, and
add a line to ENGINES. Voices for it then go in services/voices.py.
"""

from typing import Any, Protocol

from voice.log import log
from voice.priority import lower_new_threads
from voice.services.registry import load
from voice.services.voices import VoiceSpec, resolve


class TTSEngine(Protocol):
    def load_voice(self, ref: str) -> Any: ...

    def synthesize(self, text: str, voice: Any, speed: float) -> tuple[int, bytes]: ...


class TTSService(Protocol):
    voice: VoiceSpec
    speed: float

    def configure(self, voice: str | None = None, speed: float | None = None) -> None: ...

    def synthesize(self, text: str) -> tuple[int, bytes]: ...


ENGINES = {
    "kokoro": ("voice.services.kokoro.tts", "KokoroTTS"),
    "piper": ("voice.services.piper.tts", "PiperTTS"),
    "command": ("voice.services.command.tts", "CommandTTS"),
}


def engine_class(name):
    return load(ENGINES, "text-to-speech", name)


def check(name: str) -> VoiceSpec:
    """The voice a name means, if its engine can use it; raises ValueError otherwise."""
    spec = resolve(name)
    engine_class(spec.engine).check(spec.ref)
    return spec


class TTSSwitcher:
    def __init__(self, voice: str, speed: float, fallback: str = "michael", engine_opts: dict | None = None):
        """engine_opts: per engine name, the keyword arguments its constructor gets
        (provider, threads, model), e.g. {"kokoro": {"provider": "cuda"}}. Switching to a
        voice of another engine unloads the previous engine."""
        self.engines: dict[str, TTSEngine] = {}
        self.engine_opts = engine_opts or {}
        self.speed = speed
        try:
            self._use(voice)
        except ValueError as err:
            log(f"voice {voice!r}: {err}; using {fallback}")
            self._use(fallback)

    def _use(self, name):
        spec = check(name)
        # One engine loaded at a time: Kokoro alone holds about 650 MB.
        self.engines = {k: e for k, e in self.engines.items() if k == spec.engine}
        engine = self.engines.get(spec.engine)
        if engine is None:
            with lower_new_threads():
                engine = self.engines[spec.engine] = engine_class(spec.engine)(**self.engine_opts.get(spec.engine, {}))
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
