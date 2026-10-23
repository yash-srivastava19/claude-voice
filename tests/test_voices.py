"""The voice catalog: names to (engine, ref), and each engine's check. No models load."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parent.parent / "helper"
VOICED = HELPER / "voiced.py"
MODELS = Path.home() / ".local/share/voice-mode/models"
PY = Path.home() / ".local/share/voice-mode/venv/bin/python"

pytestmark = pytest.mark.skipif(not (MODELS / "voices-v1.0.bin").exists() or not PY.exists(),
                                reason="needs the installed Kokoro voices file")


@pytest.fixture
def home(tmp_path):
    (tmp_path / "voices").mkdir()
    (tmp_path / "models").symlink_to(MODELS)
    piper = tmp_path / "voices" / "amy"
    piper.mkdir()
    (piper / "amy.onnx").write_bytes(b"")
    (piper / "tokens.txt").write_text("")
    sh = tmp_path / "voices" / "mine.sh"
    sh.write_text("#!/bin/sh\ncat >/dev/null\n")
    sh.chmod(0o755)
    return tmp_path


def check(home, name):
    return subprocess.run([str(PY), str(VOICED), "--check-voice", name], capture_output=True, text=True,
                          env={**os.environ, "VOICE_HOME": str(home)})


@pytest.mark.parametrize("name", ["scarlet", "SCARLET", "michael", "af_bella", "af_heart:0.6+bf_emma:0.4",
                                  "honey", "amy", "mine"])
def test_known_voices_pass(home, name):
    assert check(home, name).returncode == 0


@pytest.mark.parametrize("name,message", [("nope", "no voice named 'nope'"), ("af_heart:x", "bad share 'x'")])
def test_unknown_voices_say_why(home, name, message):
    r = check(home, name)
    assert r.returncode == 1 and message in r.stderr


def test_engines_by_file_type(home):
    sys.path.insert(0, str(HELPER))
    os.environ["VOICE_HOME"] = str(home)
    import importlib
    import voice.config, voice.services.voices as voices
    importlib.reload(voice.config); importlib.reload(voices)
    try:
        assert voices.resolve("amy").engine == "piper"
        assert voices.resolve("mine").engine == "command"
        assert voices.resolve("scarlet") == voices.VoiceSpec("scarlet", "kokoro", "af_sky",
                                                             "light and airy, US female (Kokoro's sky)")
        assert voices.resolve("af_sky").engine == "kokoro"
    finally:
        del os.environ["VOICE_HOME"]
        importlib.reload(voice.config); importlib.reload(voices)
