"""The engine contracts, checked against every registered engine installed on this machine.

STTService.transcribe(audio) -> str
  pre   audio is 1-D float32 at 16 kHz in [-1, 1]
  post  returns a str; silence gives "" or a short non-sentence; safe to call from
        several threads at once

TTSEngine
  check(ref)                 raises ValueError for a ref it can't use, without loading a model
  load_voice(ref)            the handle synthesize takes
  synthesize(text, v, speed) -> (rate, pcm): rate > 0 and pcm 16-bit mono (even length),
                             longer text gives longer audio, a faster speed shorter audio
"""
import sys
import threading
from pathlib import Path

import numpy as np
import pytest


from voice.config import PARAKEET_MODEL, WAKE_MODEL, WHISPER_MODEL
from voice.services import stt_service, tts_service
from voice.services.voices import resolve

STT = {
    "whisper": dict(model=WHISPER_MODEL if Path(WHISPER_MODEL).exists() else WAKE_MODEL, language="en"),
    "parakeet": dict(model=PARAKEET_MODEL),
}
TTS_VOICES = {"kokoro": "scarlet", "piper": "ryan", "command": None}


def installed_stt():
    for name, opts in STT.items():
        if Path(opts["model"]).exists():
            yield name


def speech():
    """A real spoken sentence from the test set if this machine has it, else Kokoro's."""
    try:
        import soundfile as sf
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
        import data
        clip = next(c for c in data.load() if c["group"].startswith("read"))
        return sf.read(data.SET / clip["file"], dtype="float32")[0], clip["text"]
    except Exception:
        pytest.skip("no speech clip on this machine (run bench/data.py)")


def test_every_engine_is_covered():
    assert set(STT) == set(stt_service.ENGINES)
    assert set(TTS_VOICES) == set(tts_service.ENGINES)


@pytest.fixture(scope="module", params=list(installed_stt()))
def stt(request):
    return stt_service.create(request.param, **STT[request.param])


def test_stt_returns_the_words(stt):
    audio, text = speech()
    heard = stt.transcribe(audio)
    assert isinstance(heard, str)
    common = set(heard.lower().split()) & set(text.lower().split())
    assert len(common) >= len(text.split()) // 2, (heard, text)


def test_stt_silence_is_not_a_sentence(stt):
    heard = stt.transcribe(np.zeros(16000 * 2, np.float32))
    assert isinstance(heard, str) and len(heard.split()) <= 3, heard


def test_stt_is_thread_safe(stt):
    audio, _ = speech()
    results, errors = [], []

    def go():
        try:
            results.append(stt.transcribe(audio))
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=go) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(set(results)) == 1, (errors, results)


def installed_tts():
    for engine, voice in TTS_VOICES.items():
        if voice is None:
            continue
        try:
            tts_service.check(voice)
        except ValueError:
            continue
        yield engine


@pytest.fixture(scope="module", params=list(installed_tts()))
def tts(request):
    spec = resolve(TTS_VOICES[request.param])
    engine = tts_service.engine_class(spec.engine)()
    return engine, engine.load_voice(spec.ref)


@pytest.mark.parametrize("engine", list(tts_service.ENGINES))
def test_tts_check_rejects_bad_refs_without_loading(engine):
    cls = tts_service.engine_class(engine)
    with pytest.raises(ValueError):
        cls.check("/no/such/voice" if engine != "kokoro" else "xx_nobody")


def test_tts_output_is_16bit_audio(tts):
    engine, voice = tts
    rate, pcm = engine.synthesize("Okay, opening the voice log.", voice, 1.0)
    assert isinstance(rate, int) and rate > 0
    assert isinstance(pcm, bytes) and len(pcm) % 2 == 0 and len(pcm) > rate // 2


def test_tts_length_follows_text_and_speed(tts):
    engine, voice = tts
    short = len(engine.synthesize("Okay.", voice, 1.0)[1])
    long = len(engine.synthesize("Okay, so the first check is easy, and then it waits a moment.", voice, 1.0)[1])
    fast = len(engine.synthesize("Okay, so the first check is easy, and then it waits a moment.", voice, 1.5)[1])
    assert short < long and fast < long
