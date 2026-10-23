"""Paths, environment variables and tunables, read once at import."""

import os
import sys
from pathlib import Path

HOME = Path(os.environ.get("VOICE_HOME", Path.home() / ".local/share/voice-mode"))
MODELS = HOME / "models"
SOCK = Path(os.environ.get("VOICE_SOCK") or Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "claude-voice.sock")
LOG = Path(os.environ.get("VOICE_LOG") or HOME / "voiced.log")  # tests point this elsewhere

IS_MAC = sys.platform == "darwin"
# Bumped with each release. `--start` replaces a running daemon whose version or BUILD
# differs, so an updated plugin never keeps talking to the old code.
VERSION = "0.3.0"


def _build():
    import hashlib
    h = hashlib.sha1()
    for f in sorted(Path(__file__).resolve().parent.parent.rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:8]


BUILD = _build()

# Speech to text: Parakeet when its model is installed (most accurate, and faster than
# whisper on CPU), whisper otherwise. VOICE_STT forces one (see services/stt_service.py).
PARAKEET_MODEL = os.environ.get("VOICE_PARAKEET_MODEL", str(MODELS / "parakeet-tdt-0.6b-v2-int8"))
WHISPER_MODEL = os.environ.get("VOICE_WHISPER_MODEL", str(MODELS / "ggml-base.en.bin"))
STT_ENGINE = os.environ.get("VOICE_STT", "") or ("parakeet" if Path(PARAKEET_MODEL).is_dir() else "whisper")
# The small model that listens for "hey Claude" while asleep.
WAKE_MODEL = os.environ.get("VOICE_WAKE_MODEL", str(MODELS / "ggml-tiny.en.bin"))

# Text to speech. The mod sends the voice and speed it is configured with (POST /config);
# these are the starting values until it does.
KOKORO_MODEL = os.environ.get("VOICE_KOKORO_MODEL", str(MODELS / "kokoro-v1.0.onnx"))
KOKORO_VOICES = MODELS / "voices-v1.0.bin"
KOKORO_VOICE = os.environ.get("VOICE_KOKORO_VOICE", "michael")
KOKORO_SPEED = float(os.environ.get("VOICE_KOKORO_SPEED", "1.05"))
# Your own voices: <name>.npy or <name>.pt (a Kokoro voice) or an executable
# <name>.sh (any TTS: text on stdin, a WAV on stdout).
USER_VOICES = HOME / "voices"
# Long sentences are spoken in clause-sized chunks, so the first sound starts sooner.
CHUNK_WORDS = 12
# Priority (nice) of text to speech's threads, so your words are transcribed first; 0 is off.
TTS_NICE = int(os.environ.get("VOICE_TTS_NICE", "10"))

# Audio devices. A PipeWire source name to record from instead of the default mic, such as
# the laptop mic while Bluetooth headphones stay in their high-quality (A2DP) profile.
MIC_SOURCE = os.environ.get("VOICE_MIC_SOURCE", "")
SPEAKER_SINK = os.environ.get("VOICE_SPEAKER_SINK", "")
MIC_CMD = os.environ.get("VOICE_MIC_CMD", "")  # override for tests: any 16 kHz mono s16 stream
# Echo-cancelled devices from ~/.config/pipewire/pipewire.conf.d/60-claude-voice-echo-cancel.conf.
# When they exist, and nothing above overrides them, the daemon plays into the sink and
# records from the source, so PipeWire subtracts its own voice from the mic.
AEC_SOURCE, AEC_SINK = "claude_voice_mic", "claude_voice_out"
PLAY_BLOCK_S = 0.05  # playback is written in blocks this long
PLAY_LEAD_S = 0.25  # at most this much audio sits buffered ahead, so pause and stop are quick

# Turn taking.
START_FRAMES = 3  # ~100 ms of speech opens an utterance
BARGE_FRAMES = 10  # ~320 ms of confident speech pauses playback until the transcript decides
END_SILENCE_S = float(os.environ.get("VOICE_END_SILENCE_S", "0.6"))
PREROLL_FRAMES = 10
MAX_UTTERANCE_S = 30
THRESHOLD = 0.5
BARGE_THRESHOLD = 0.85  # stricter while speaking, so our own voice rarely counts

# How the daemon keeps its own voice out of the mic, chosen once at start:
#   aec   the mic and speaker are an echo-cancelled pair, so you can talk over Claude
#   gate  no echo cancellation (speakers on macOS, Linux without the PipeWire config):
#         the mic is ignored while Claude speaks and for ECHO_TAIL_S after
#   open  nothing is held back; for headphones, where the mic never hears the speaker
# VOICE_ECHO forces one; unset, it is "aec" when the echo-cancelled devices are in use
# and "gate" otherwise.
ECHO_MODES = ("aec", "gate", "open")
ECHO_MODE = os.environ.get("VOICE_ECHO", "").strip().lower()
# Speakers keep sounding a little after the last block is written, and a room echoes.
ECHO_TAIL_S = float(os.environ.get("VOICE_ECHO_TAIL_S", "0.5"))

# Sleep and wake: after SLEEP_AFTER_S with no conversation, the daemon stops passing
# speech on and listens only for the wake phrase ("hey Claude"), checked by a small
# whisper model kept in memory. A typed prompt or /wake also wakes it.
SLEEP_AFTER_S = float(os.environ.get("VOICE_SLEEP_AFTER_S", "120"))
WAKE_WINDOW_S = 2.5  # the wake phrase opens a segment; only its start is checked
WAKE_MAX_SEGMENT_S = 12.0  # longer segments while asleep are background talk

WATCHDOG_S = 20

# In power-saver mode Kokoro runs slower than real time and every sentence ends in a
# gap, so the daemon holds the balanced profile while it runs. The profile it found is
# kept in this file until it is restored, so a crash cannot strand the laptop on balanced.
POWER_FILE = HOME / "power-restore"
VOICE_POWER = os.environ.get("VOICE_POWER", "balanced")  # "off" leaves the profile alone
