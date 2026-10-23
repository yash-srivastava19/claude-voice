"""The whole daemon, end to end, through its socket: fake mic in, null sink out.

Slow (loads the real models) and Linux-only (PulseAudio null sinks). Run with
    VOICE_DAEMON=path/to/voiced.py pytest tests/test_pipeline.py
to check a given version; it defaults to this checkout.
"""
import json, os, shutil, socket, subprocess, sys, time, wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DAEMON = Path(os.environ.get("VOICE_DAEMON", ROOT / "helper/voiced.py"))
VENV_PY = Path.home() / ".local/share/voice-mode/venv/bin/python"
MODELS = Path.home() / ".local/share/voice-mode/models"
FIXTURES = Path(os.environ.get("VOICE_TEST_CACHE", Path.home() / ".cache/claude-voice-tests"))
LINES = {
    "question": "Can you check why the voice log shows a slow transcription?",
    "backchannel": "Uh-huh, right.",
    "interrupt": "Wait, why is it slow?",
    "echo": "Here is a long explanation of how the gate works and why the first word is kept.",
}
LONG = ("Here is a long explanation of how the gate works, how the tail is recorded, and why "
        "the first word of a quick reply is kept now, so you can follow along.")

pytestmark = pytest.mark.skipif(not shutil.which("pactl") or not VENV_PY.exists(),
                                reason="needs PulseAudio/PipeWire and the installed voice venv")


def clips():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    missing = [k for k in LINES if not (FIXTURES / f"{k}.raw").exists()]
    if missing:
        from kokoro_onnx import Kokoro
        k = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
        for name in missing:
            pcm, rate = k.create(LINES[name], voice="am_michael", speed=1.0, lang="en-us")
            idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * 16000 / rate))
            pcm = np.interp(idx, np.arange(len(pcm)), pcm)
            (FIXTURES / f"{name}.raw").write_bytes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())


class Daemon:
    def __init__(self, tmp, echo="open"):
        self.tmp, self.ctl = tmp, tmp / "mic"
        self.ctl.mkdir()
        self.sock, self.log = tmp / "v.sock", tmp / "v.log"
        self.modules = [subprocess.run(["pactl", "load-module", "module-null-sink", f"sink_name={n}"],
                                       capture_output=True, text=True).stdout.strip() for n in ("vtest_spk",)]
        env = {**os.environ, "VOICE_SOCK": str(self.sock), "VOICE_LOG": str(self.log),
               "VOICE_SPEAKER_SINK": "vtest_spk", "VOICE_ECHO": echo, "VOICE_POWER": "off",
               "VOICE_SLEEP_AFTER_S": "0",
               "VOICE_MIC_CMD": f"{sys.executable} {ROOT / 'tests/fake_mic.py'} {self.ctl}"}
        subprocess.run([str(VENV_PY), str(DAEMON), "--start"], env=env, check=True, capture_output=True)
        self.events()

    def call(self, method, path, body=None):
        data = json.dumps(body or {}).encode()
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(5)
            s.connect(str(self.sock))
            s.sendall(f"{method} {path} HTTP/1.0\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data)
            reply = b""
            while chunk := s.recv(65536):
                reply += chunk
        try:
            return json.loads(reply.split(b"\r\n\r\n", 1)[1])
        except (IndexError, ValueError):
            log = self.log.read_text()[-2000:] if self.log.exists() else "(no log)"
            raise AssertionError(f"{method} {path}: bad reply {reply[:300]!r}\n--- daemon log ---\n{log}")

    def events(self):
        return self.call("GET", "/events")["events"]

    def speak_into_mic(self, name):
        (self.ctl / f"{time.monotonic_ns()}.raw").write_bytes((FIXTURES / f"{name}.raw").read_bytes())

    def wait_for(self, pred, timeout=25):
        seen, end = [], time.monotonic() + timeout
        while time.monotonic() < end:
            seen += self.events()
            if pred(seen):
                return seen
            time.sleep(0.25)
        raise AssertionError(f"timed out; events: {[e['kind'] for e in seen]}\n{self.log.read_text()[-1500:]}")

    def close(self):
        try:
            self.call("POST", "/shutdown")
        except OSError:
            pass
        time.sleep(1)
        for m in self.modules:
            subprocess.run(["pactl", "unload-module", m], capture_output=True)


kinds = lambda evs: [e["kind"] for e in evs]


@pytest.fixture
def daemon(tmp_path):
    clips()
    d = Daemon(tmp_path)
    yield d
    d.close()


def test_utterance(daemon):
    daemon.speak_into_mic("question")
    evs = daemon.wait_for(lambda e: "utterance" in kinds(e))
    text = next(e["text"] for e in evs if e["kind"] == "utterance").lower()
    assert "voice log" in text and "hearing" in kinds(evs)


def test_say_then_idle(daemon):
    daemon.call("POST", "/say", {"text": "Okay, opening the voice log.", "gen": 0})
    evs = daemon.wait_for(lambda e: "idle" in kinds(e))
    assert kinds(evs).index("speaking") < kinds(evs).index("idle")


def test_backchannel_does_not_interrupt(daemon):
    daemon.call("POST", "/say", {"text": LONG, "gen": 0})
    daemon.wait_for(lambda e: "speaking" in kinds(e))
    daemon.speak_into_mic("backchannel")
    evs = daemon.wait_for(lambda e: any(x["kind"] == "dropped" for x in e) or "bargein" in kinds(e))
    assert "bargein" not in kinds(evs)
    assert any(e["kind"] == "dropped" and e.get("reason") == "backchannel" for e in evs)


def test_interruption_barges_in(daemon):
    daemon.call("POST", "/say", {"text": LONG, "gen": 0})
    daemon.wait_for(lambda e: "speaking" in kinds(e))
    daemon.speak_into_mic("interrupt")
    evs = daemon.wait_for(lambda e: "utterance" in kinds(e))
    assert "bargein" in kinds(evs)


def test_echo_is_dropped(daemon):
    daemon.call("POST", "/say", {"text": LINES["echo"], "gen": 0})
    daemon.wait_for(lambda e: "speaking" in kinds(e))
    daemon.speak_into_mic("echo")
    evs = daemon.wait_for(lambda e: "dropped" in kinds(e) or "utterance" in kinds(e))
    assert any(e["kind"] == "dropped" and e.get("reason") == "echo" for e in evs), evs


def test_mic_reconnects(daemon):
    subprocess.run(["pkill", "-f", f"fake_mic.py {daemon.ctl}"])
    evs = daemon.wait_for(lambda e: any(x["kind"] == "mic" and x.get("ok") for x in e), timeout=15)
    assert [e.get("ok") for e in evs if e["kind"] == "mic"][:2] == [False, True]
    daemon.speak_into_mic("question")
    daemon.wait_for(lambda e: "utterance" in kinds(e))
