"""Voice daemon for the Claude Code `voice` mod.

Listens on the mic, detects turns with Silero VAD, transcribes with whisper.cpp,
and speaks text with Kokoro. The mod talks to it over HTTP on a Unix socket:

  GET  /events          drain queued events (JSON list); also the watchdog ping
  POST /say  {text,gen} queue a sentence; dropped if gen is older than current
  POST /flush {gen}     stop speaking, drop queued sentences older than gen
  POST /listen {on}     pause or resume the mic
  POST /config {voice,speed}  change the voice or speaking rate (HTTP 400 if unknown)
  POST /shutdown        exit

`voiced.py --voices` prints the voices as JSON; `voiced.py --check-voice NAME`
exits 0 if NAME is a voice. Neither loads a model.

`voiced.py --start` starts the daemon if no live one owns the socket, then exits.
The daemon exits on its own when nothing polls /events for WATCHDOG_S seconds,
so a closed session or a reloaded mod never leaves it holding the mic.
"""

import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path(os.environ.get("VOICE_HOME", Path.home() / ".local/share/voice-mode"))
MODELS = HOME / "models"
SOCK = Path(os.environ.get("VOICE_SOCK") or Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "claude-voice.sock")
LOG = Path(os.environ.get("VOICE_LOG") or HOME / "voiced.log")  # tests point this elsewhere

WHISPER_MODEL = os.environ.get("VOICE_WHISPER_MODEL", str(MODELS / "ggml-base.en.bin"))
IS_MAC = sys.platform == "darwin"
# Bumped with each release: `--start` replaces a running daemon of another version.
VERSION = "0.2.0"
KOKORO_MODEL = os.environ.get("VOICE_KOKORO_MODEL", str(MODELS / "kokoro-v1.0.onnx"))
KOKORO_VOICES = MODELS / "voices-v1.0.bin"
# The mod sends the voice and speed it is configured with (POST /config); these
# are the starting values until it does.
KOKORO_VOICE = os.environ.get("VOICE_KOKORO_VOICE", "michael")
KOKORO_SPEED = float(os.environ.get("VOICE_KOKORO_SPEED", "1.05"))
# Your own voices: <name>.npy or <name>.pt (a Kokoro voice) or an executable
# <name>.sh (any TTS: text on stdin, a WAV on stdout).
USER_VOICES = HOME / "voices"

# Picked from Kokoro's best-rated English voices. A "+" joins voices into a mix,
# each weighted by its ":<share>".
PRESETS = {
    "heart": ("af_heart", "warm, US female"),
    "bella": ("af_bella", "bright and expressive, US female"),
    "nicole": ("af_nicole", "soft, close to the mic, US female"),
    "emma": ("bf_emma", "calm, British female"),
    "michael": ("am_michael", "steady, US male"),
    "fenrir": ("am_fenrir", "deep, US male"),
    "puck": ("am_puck", "lively, US male"),
    "george": ("bm_george", "warm, British male"),
    "fable": ("bm_fable", "storyteller, British male"),
    "honey": ("af_heart:0.6+af_bella:0.4", "mix of heart and bella"),
    "velvet": ("af_nicole:0.5+af_heart:0.5", "mix of nicole and heart"),
}
# Kokoro's prefix: language, then f/m. Only English is wired to the transcriber.
LANGS = {"a": "en-us", "b": "en-gb"}
# A PipeWire source name to record from instead of the default mic, such as the laptop mic
# while Bluetooth headphones stay in their high-quality (A2DP) profile.
MIC_SOURCE = os.environ.get("VOICE_MIC_SOURCE", "")
SPEAKER_SINK = os.environ.get("VOICE_SPEAKER_SINK", "")
# Echo-cancelled devices from ~/.config/pipewire/pipewire.conf.d/60-claude-voice-echo-cancel.conf.
# When they exist, and nothing above overrides them, the daemon plays into the sink and
# records from the source, so PipeWire subtracts its own voice from the mic.
AEC_SOURCE, AEC_SINK = "claude_voice_mic", "claude_voice_out"
# Long sentences are spoken in clause-sized chunks, so the first sound starts sooner.
CHUNK_WORDS = 12

RATE = 16000
FRAME = 512  # Silero v5 frame at 16 kHz, 32 ms
FRAME_S = FRAME / RATE
START_FRAMES = 3  # ~100 ms of speech opens an utterance
BARGE_FRAMES = 10  # ~320 ms of confident speech pauses playback until the transcript decides
END_SILENCE_S = float(os.environ.get("VOICE_END_SILENCE_S", "0.6"))
PREROLL_FRAMES = 10
MAX_UTTERANCE_S = 30
THRESHOLD = 0.5
BARGE_THRESHOLD = 0.85  # stricter while speaking, so our own voice rarely counts
WATCHDOG_S = 20
PLAY_BLOCK_S = 0.05  # playback is written in blocks this long
PLAY_LEAD_S = 0.25  # at most this much audio sits buffered ahead, so pause and stop are quick

# Things whisper "hears" in silence or noise.
HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "so", "uh", "um", "hmm", "bye bye", "thank you very much", "thanks a lot",
    "see you in the next video", "ill see you in the next video", "i will see you in the next video",
    "please subscribe", "like and subscribe", "love you", "i love you",
}

# Sleep and wake: after SLEEP_AFTER_S with no conversation, the daemon stops passing
# speech on and listens only for the wake phrase ("hey Claude"), checked by a small
# whisper model kept in memory. A typed prompt or /wake also wakes it.
SLEEP_AFTER_S = float(os.environ.get("VOICE_SLEEP_AFTER_S", "120"))
WAKE_MODEL = os.environ.get("VOICE_WAKE_MODEL", str(MODELS / "ggml-tiny.en.bin"))
WAKE_WINDOW_S = 2.5  # the wake phrase opens a segment; only its start is checked
WAKE_MAX_SEGMENT_S = 12.0  # longer segments while asleep are background talk
WAKE_PHRASE = re.compile(
    r"^\s*(?:(?:hey|hay|hi|okay|ok|yo|eh|a)[\s,-]+)?"
    r"(?:claude|claud|clod|clord|clode|clawed|clyde|cloud|clone)\b[\s,.!?]*",
    re.IGNORECASE,
)
# How the small model mishears "hey Claude" through a headset mic: "A-Clord", "a clone",
# "OK cloud". A greeting-like word plus a cl-word, or a cl-word alone, opens the segment.
GREETING_LIKE = {"hey", "hay", "hi", "okay", "ok", "yo", "eh", "a", "hei", "heh"}
CLAUDE_LIKE = re.compile(r"^[ck]l(?:a|o|au|aw|ou|or|y)[a-z]{0,3}$")
NOT_CLAUDE = {"close", "closed", "clean", "clear", "class", "clock", "cloth", "clue", "claim",
              "clap", "clip", "club", "claws", "clay", "clan", "clam", "clots"}


def heard_wake(text: str) -> bool:
    """Lenient wake test for the small model's transcript of a segment's opening."""
    tokens = re.findall(r"[a-z]+", text.lower())
    def claude_like(token):
        return bool(CLAUDE_LIKE.match(token)) and token not in NOT_CLAUDE
    if tokens and claude_like(tokens[0]):
        return True
    return len(tokens) > 1 and tokens[0] in GREETING_LIKE and claude_like(tokens[1])


def strip_wake(text: str):
    """Returns (is_wake, text after the wake phrase) for a transcript."""
    m = WAKE_PHRASE.match(text)
    if not m:
        return False, text
    return True, text[m.end():].strip()


ECHO_WINDOW_S = 15
ECHO_OVERLAP = 0.6
# Short interruptions that always get through, even when Claude just said the same word.
STOP_PHRASES = {"stop", "wait", "hold on", "cancel", "never mind", "shut up"}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def is_echo(transcript: str, recent: list[str], during_speech: bool = False) -> bool:
    """True when a transcript is mostly words the daemon itself just spoke.

    Speakers feed Claude's voice back into the mic, more so without echo cancellation;
    this keeps those transcripts from going back to Claude as your words. One or two
    words heard while Claude was talking count as echo when Claude said all of them.
    """
    heard = [stem(w) for w in words(transcript)]
    if not heard:
        return False
    spoken = {stem(w) for sentence in recent for w in words(sentence)}
    if len(heard) < 3:
        if not during_speech or " ".join(words(transcript)) in STOP_PHRASES:
            return False
        return all(w in spoken for w in heard)
    return sum(w in spoken for w in heard) / len(heard) >= ECHO_OVERLAP


def pw_node_exists(kind: str, name: str) -> bool:
    """True when PipeWire has a sink or source (kind "sinks" or "sources") with this name."""
    try:
        out = subprocess.run(["pactl", "list", "short", kind],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.split("\t")[1:2] == [name] for line in out.splitlines())


def chunk_sentence(text: str) -> list[str]:
    """Splits a long sentence at clause punctuation into chunks of at least 4 words."""
    if len(text.split()) <= CHUNK_WORDS:
        return [text]
    parts = re.split(r"(?<=[,;:])\s+", text)
    chunks, current = [], ""
    for part in parts:
        current = f"{current} {part}".strip()
        if len(current.split()) >= 4:
            chunks.append(current)
            current = ""
    if current:
        if chunks:
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)
    return chunks


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, file=sys.stderr, flush=True)


def clean_transcript(text: str) -> str:
    """Strip whisper's non-speech tags and drop known hallucinations."""
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    bare = re.sub(r"[^a-z ]", "", text.lower()).strip()
    if not bare or bare in HALLUCINATIONS:
        return ""
    return text


def kokoro_ids():
    import numpy as np
    with np.load(KOKORO_VOICES) as z:
        return sorted(k for k in z.files if k[:1] in LANGS)


def user_voices():
    if not USER_VOICES.is_dir():
        return {}
    found = {}
    for f in sorted(USER_VOICES.iterdir()):
        if f.suffix in (".npy", ".pt") or (f.suffix == ".sh" and os.access(f, os.X_OK)):
            found.setdefault(f.stem, f)
    return found


def list_voices():
    return {
        "presets": [{"name": n, "voice": v, "about": a} for n, (v, a) in PRESETS.items()],
        "kokoro": kokoro_ids(),
        "yours": [{"name": n, "file": str(f)} for n, f in user_voices().items()],
    }


class Voice:
    """A resolved voice: a Kokoro style (a built-in, a mix or a file) or a command."""

    def __init__(self, name, style=None, lang="en-us", command=None):
        self.name, self.style, self.lang, self.command = name, style, lang, command


def resolve_voice(name, np, styles=None):
    """Turns a voice name into a Voice; raises ValueError naming what is wrong.

    Accepts a preset ("heart"), a Kokoro id ("af_heart"), a mix
    ("af_heart:0.6+bf_emma:0.4") or one of your voices in USER_VOICES.
    """
    name = name.strip()
    spec = PRESETS.get(name.lower(), (name, ""))[0]
    mine = user_voices().get(name)
    if mine is not None:
        if mine.suffix == ".sh":
            return Voice(name, command=str(mine))
        style = np.load(mine) if mine.suffix == ".npy" else load_pt(mine)
        style = np.asarray(style, dtype=np.float32).reshape(-1, 1, 256)
        return Voice(name, style=style)
    if styles is None:
        styles = np.load(KOKORO_VOICES)
    parts, total, mix = [], 0.0, None
    for part in spec.split("+"):
        vid, _, share = part.strip().partition(":")
        if vid not in styles.files or vid[:1] not in LANGS:
            raise ValueError(f"no voice named {vid!r}; /talk voices lists them")
        try:
            weight = float(share) if share else 1.0
        except ValueError:
            raise ValueError(f"bad share {share!r} in {part!r}; write it like af_heart:0.6") from None
        parts.append(vid)
        total += weight
        mix = styles[vid] * weight if mix is None else mix + styles[vid] * weight
    if total <= 0:
        raise ValueError("the shares in a mix must add up to more than 0")
    return Voice(name, style=(mix / total).astype(np.float32), lang=LANGS[parts[0][0]])


def load_pt(path):
    try:
        import torch
    except ImportError:
        raise ValueError(f"{path.name} is a PyTorch file; convert it once to .npy (see the README)") from None
    return torch.load(path, map_location="cpu").numpy()


class KokoroTTS:
    def __init__(self, np):
        from kokoro_onnx import Kokoro
        self.np = np
        self.model = Kokoro(KOKORO_MODEL, str(KOKORO_VOICES))
        self.styles = np.load(KOKORO_VOICES)
        self.speed = KOKORO_SPEED
        try:
            self.voice = resolve_voice(KOKORO_VOICE, np, self.styles)
        except ValueError as err:
            log(f"voice {KOKORO_VOICE!r}: {err}; using michael")
            self.voice = resolve_voice("michael", np, self.styles)
        # Warm up on a full sentence: the first real call is otherwise about twice as slow.
        for _ in range(2):
            self.model.create("Sure, so here is how the whole thing fits together.",
                              voice=self.styles["am_michael"], lang="en-us")

    def configure(self, voice=None, speed=None):
        """Applies a new voice and/or speed; raises ValueError if either is bad."""
        if speed is not None:
            speed = float(speed)
            if not 0.5 <= speed <= 2.0:
                raise ValueError("speed must be between 0.5 and 2.0")
        if voice is not None and voice != self.voice.name:
            self.voice = resolve_voice(voice, self.np, self.styles)
            log(f"voice: {self.voice.name}")
        if speed is not None:
            self.speed = speed

    def synthesize(self, text):
        v = self.voice
        if v.command:
            return self.run_command(v.command, text)
        audio, rate = self.model.create(text, voice=v.style, speed=self.speed, lang=v.lang)
        return rate, (self.np.clip(audio, -1, 1) * 32767).astype(self.np.int16).tobytes()

    def run_command(self, command, text):
        import io
        import wave
        res = subprocess.run([command], input=text.encode(), capture_output=True, timeout=60,
                             env={**os.environ, "VOICE_SPEED": str(self.speed)})
        if res.returncode != 0:
            raise RuntimeError(f"{command} exited {res.returncode}: {res.stderr.decode()[-200:]}")
        with wave.open(io.BytesIO(res.stdout)) as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if width != 2:
            raise RuntimeError(f"{command}: write 16-bit WAV (got {8 * width}-bit)")
        pcm = self.np.frombuffer(frames, dtype=self.np.int16)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(self.np.int16)
        return rate, pcm.tobytes()


# In power-saver mode Kokoro runs slower than real time and every sentence ends in a
# gap, so the daemon holds the balanced profile while it runs. The profile it found is
# kept in this file until it is restored, so a crash cannot strand the laptop on balanced.
POWER_FILE = HOME / "power-restore"
VOICE_POWER = os.environ.get("VOICE_POWER", "balanced")  # "off" leaves the profile alone


def power_profile(set_to=None):
    cmd = ["powerprofilesctl", "set", set_to] if set_to else ["powerprofilesctl", "get"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return res.stdout.strip() if res.returncode == 0 else ""


def hold_power():
    if VOICE_POWER == "off":
        return
    now = power_profile()
    if now == "power-saver":
        if not POWER_FILE.exists():
            POWER_FILE.write_text(now)
        power_profile(VOICE_POWER)
        log(f"power: {now} -> {power_profile() or 'unchanged'} while voice is on")


def release_power():
    if not POWER_FILE.exists():
        return
    before = POWER_FILE.read_text().strip()
    if power_profile() == VOICE_POWER:  # left alone if you changed it yourself meanwhile
        power_profile(before)
        log(f"power: restored {before}")
    POWER_FILE.unlink(missing_ok=True)


class Player:
    """One long-lived playback stream, fed in short paced blocks.

    A player process per sentence clicked and left a gap while the device opened.
    Pacing keeps at most PLAY_LEAD_S buffered, so pause and kill take effect at once.
    """

    def __init__(self, sink):
        self.sink = sink
        self.proc = None
        self.rate = 0
        self.until = 0.0  # monotonic time at which everything written has played
        self.lock = threading.RLock()  # _open calls kill while holding it
        self.resumed = threading.Event()  # cleared while a possible barge-in is checked
        self.resumed.set()

    def _open(self, rate):
        if self.proc and self.proc.poll() is None and self.rate == rate:
            return self.proc
        self.kill()
        if IS_MAC:
            cmd = ["play", "-q", "-t", "raw", "-r", str(rate), "-e", "signed", "-b", "16", "-c", "1", "-"]
        elif self.sink:
            cmd = ["pw-cat", "--playback", "--raw", "--target", self.sink, "--rate", str(rate),
                   "--channels", "1", "--format", "s16", "-"]
        else:
            cmd = ["aplay", "-q", "-f", "S16_LE", "-r", str(rate), "-c", "1", "-t", "raw"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.rate = rate
        return self.proc

    def write(self, pcm, rate, cancelled):
        """Plays pcm. Returns False if cancelled() turned true or the stream died first."""
        with self.lock:
            proc = self._open(rate)
        step = int(rate * PLAY_BLOCK_S) * 2
        for i in range(0, len(pcm), step):
            while True:
                if cancelled():
                    return False
                if not self.resumed.is_set():
                    self.resumed.wait(0.05)
                    continue
                ahead = self.until - time.monotonic()
                if ahead <= PLAY_LEAD_S:
                    break
                time.sleep(min(ahead - PLAY_LEAD_S, 0.05))
            block = pcm[i:i + step]
            try:
                proc.stdin.write(block)
                proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                return False
            self.until = max(self.until, time.monotonic()) + len(block) / 2 / rate
        return True

    def busy(self):
        return self.until > time.monotonic()

    def kill(self):
        with self.lock:
            proc, self.proc = self.proc, None
        self.until = 0.0
        if proc and proc.poll() is None:
            proc.kill()


class Daemon:
    def __init__(self):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.vad = ort.InferenceSession(str(MODELS / "silero_vad.onnx"), sess_options=opts)
        self.tts = KokoroTTS(np)
        log("tts:", f"kokoro ({self.tts.voice.name}, speed {self.tts.speed})")
        from pywhispercpp.model import Model

        def whisper(path, threads):
            return Model(path, n_threads=threads, print_progress=False,
                         print_realtime=False, redirect_whispercpp_logs_to=None)

        self.stt = whisper(WHISPER_MODEL, min(8, os.cpu_count() or 4))
        self.stt_lock = threading.Lock()
        self.wake_model = None
        if SLEEP_AFTER_S > 0 and Path(WAKE_MODEL).exists():
            try:
                self.wake_model = whisper(WAKE_MODEL, 4)
                log(f"wake: sleeps after {SLEEP_AFTER_S:.0f}s of quiet; wakes on 'hey Claude'")
            except Exception as err:
                log("wake model unavailable, auto-sleep off:", repr(err))
        else:
            log("auto-sleep off")

        self.events = []
        self.events_lock = threading.Lock()
        self.last_poll = time.monotonic()

        self.gen = 0
        self.say_q = queue.Queue()  # (gen, text) waiting to be synthesized
        self.audio_q = queue.Queue()  # (gen, text, rate, pcm) waiting to be played
        self.busy = 0  # sentences taken off say_q and not yet played or dropped
        self.busy_lock = threading.Lock()
        self.mic_source = MIC_SOURCE or (AEC_SOURCE if pw_node_exists("sources", AEC_SOURCE) else "")
        sink = SPEAKER_SINK or (AEC_SINK if pw_node_exists("sinks", AEC_SINK) else "")
        self.player = Player(sink)
        log(f"audio: mic {self.mic_source or 'default'}, speaker {sink or 'default'}")
        self.chunked_gen = -1  # only a reply's first sentence is split, so it starts sooner
        self.speaking = False
        self.last_played_gen = -1
        self.recent_spoken = []  # (monotonic time, sentence), for echo detection
        self.listening = True
        self.awake = True
        self.last_activity = time.monotonic()
        self.wake_lock = threading.Lock()
        self.stopping = threading.Event()

    # events -------------------------------------------------------------

    def emit(self, kind, **data):
        with self.events_lock:
            self.events.append({"kind": kind, "at": time.time(), **data})

    def drain(self):
        self.last_poll = time.monotonic()
        with self.events_lock:
            out, self.events = self.events, []
        return out

    # speech out ---------------------------------------------------------
    #
    # Two threads: synth_loop turns sentences into audio while play_loop plays
    # the previous one, so a slow voice model costs latency on the first
    # sentence only.

    def say(self, text, gen):
        if gen < self.gen:
            log(f"say dropped: gen {gen} < {self.gen}")
            return
        self.last_activity = time.monotonic()
        # Clause-sized pieces each get their own intonation, which sounds choppy; only the
        # first sentence of a reply is split, so the first sound comes sooner.
        chunks = chunk_sentence(text) if gen != self.chunked_gen else [text]
        self.chunked_gen = gen
        for chunk in chunks:
            with self.busy_lock:
                self.busy += 1
            self.say_q.put((gen, chunk))

    def done_one(self):
        with self.busy_lock:
            self.busy = max(0, self.busy - 1)

    def flush(self, gen, why):
        self.gen = max(self.gen, gen)
        dropped = 0
        for q in (self.say_q, self.audio_q):
            while True:
                try:
                    q.get_nowait()
                    self.done_one()
                    dropped += 1
                except queue.Empty:
                    break
        was_playing = self.player.busy()
        self.stop_player()
        log(f"flush ({why}) -> gen {self.gen}: dropped {dropped} queued, {'cut' if was_playing else 'no'} playback")

    def stop_player(self):
        self.player.kill()
        self.player.resumed.set()

    def synth_loop(self):
        while not self.stopping.is_set():
            try:
                gen, text = self.say_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen < self.gen:
                self.done_one()
                continue
            try:
                t = time.monotonic()
                rate, pcm = self.tts.synthesize(text)
                if rate:
                    log(f"tts {time.monotonic() - t:.2f}s for {len(pcm) / 2 / rate:.1f}s audio")
            except Exception as err:  # keep the loop alive on a bad sentence
                log("tts error:", repr(err))
                rate, pcm = 0, b""
            if not pcm or gen < self.gen:
                self.done_one()
                continue
            self.audio_q.put((gen, text, rate, pcm))

    def play_loop(self):
        while not self.stopping.is_set():
            try:
                gen, text, rate, pcm = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                idle = self.busy == 0 and not self.player.busy() and self.player.resumed.is_set()
                if self.speaking and idle:
                    self.speaking = False
                    self.emit("idle")
                continue
            if gen >= self.gen:
                if not self.speaking:
                    self.speaking = True
                    self.emit("speaking")
                if gen != self.last_played_gen:
                    self.last_played_gen = gen
                    log(f"first audio for gen {gen}")
                self.play(gen, text, rate, pcm)
            self.done_one()

    def play(self, gen, text, rate, pcm):
        now = time.monotonic()
        self.recent_spoken = [(t, s) for t, s in self.recent_spoken if now - t < ECHO_WINDOW_S][-9:]
        self.recent_spoken.append((now, text))
        done = self.player.write(pcm, rate, lambda: gen < self.gen or self.stopping.is_set())
        log(f"queued {len(pcm) / 2 / rate:.2f}s of audio in {time.monotonic() - now:.2f}s"
            + ("" if done else ", cut"))

    # sleep and wake -----------------------------------------------------

    def sleep(self, why):
        if self.awake and self.wake_model is not None:
            self.awake = False
            log(f"asleep ({why})")
            self.emit("sleep", why=why)

    def wake(self, why):
        self.last_activity = time.monotonic()
        if not self.awake:
            self.awake = True
            log(f"awake ({why})")
            self.emit("wake", why=why)

    def check_wake(self, audio):
        """While asleep: wakes on a segment that opens with the wake phrase."""
        speech_s = len(audio) / 2 / RATE
        if speech_s < 0.4 or speech_s > WAKE_MAX_SEGMENT_S:
            return
        head = audio[: int(WAKE_WINDOW_S * RATE) * 2]
        x = self.np.frombuffer(head, dtype=self.np.int16).astype(self.np.float32) / 32768.0
        t = time.monotonic()
        try:
            with self.wake_lock:
                segs = self.wake_model.transcribe(
                    x, audio_ctx=min(1500, int(len(x) / RATE * 50) + 64),
                    single_segment=True, no_context=True, max_tokens=10,
                    initial_prompt="Hey Claude.",  # nudges a muffled "hey Claude" toward its spelling
                )
            heard = " ".join(seg.text.strip() for seg in segs)
        except Exception as err:
            log("wake check error:", repr(err))
            return
        is_wake = heard_wake(heard)
        log(f"wake check {time.monotonic() - t:.2f}s: {heard!r}{' -> wake' if is_wake else ''}")
        if not is_wake:
            return
        self.wake("wake phrase")
        # "Hey Claude, what does this do?" carries a request: transcribe it properly.
        if speech_s > WAKE_WINDOW_S - 0.5:
            self.transcribe(audio, False, strip_wake_phrase=True)

    # speech in ----------------------------------------------------------

    def listen_loop(self):
        np = self.np
        mic = os.environ.get("VOICE_MIC_CMD")  # override for tests: any 16 kHz mono s16 stream
        if mic:
            cmd = mic.split()
        elif IS_MAC:
            cmd = ["rec", "-q", "-t", "raw", "-r", str(RATE), "-e", "signed", "-b", "16", "-c", "1", "-"]
        elif self.mic_source:
            cmd = ["pw-cat", "--record", "--raw", "--target", self.mic_source,
                   "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"]
        else:
            cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"]
        log("mic:", " ".join(cmd[:4]), self.mic_source or "default source")
        rec = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        sr = np.array(RATE, dtype=np.int64)

        preroll, voiced = [], []
        in_speech = False
        hot = 0  # consecutive frames over the threshold
        silence_s = 0.0
        barged = False

        while not self.stopping.is_set():
            raw = rec.stdout.read(FRAME * 2)
            if len(raw) < FRAME * 2:
                log("mic stream ended")
                break
            pcm = np.frombuffer(raw, dtype=np.int16)
            x = (pcm.astype(np.float32) / 32768.0).reshape(1, -1)
            out, state = self.vad.run(None, {"input": np.concatenate([context, x], axis=1),
                                              "state": state, "sr": sr})
            context = x[:, -64:]
            prob = float(out[0][0])

            if not self.listening:
                in_speech, voiced, hot = False, [], 0
                self.player.resumed.set()  # a paused reply must not stay paused with the mic off
                continue

            threshold = BARGE_THRESHOLD if self.speaking else THRESHOLD
            need = BARGE_FRAMES if self.speaking else START_FRAMES
            hot = hot + 1 if prob >= threshold else 0

            if not in_speech:
                preroll = (preroll + [raw])[-PREROLL_FRAMES:]
                if hot >= need:
                    in_speech, voiced, silence_s, barged = True, list(preroll), 0.0, False
                    if not self.awake:
                        continue  # asleep: no barge-in, no events, only a wake check at the end
                    if self.speaking:
                        # Pause rather than stop: if the transcript turns out to be our own
                        # echo or noise, the reply carries on where it left off.
                        barged = True
                        self.player.resumed.clear()
                    log("hearing" + (" (paused for a possible barge-in)" if barged else ""))
                    self.emit("hearing")
                continue

            voiced.append(raw)
            silence_s = 0.0 if prob >= THRESHOLD else silence_s + FRAME_S
            too_long = len(voiced) * FRAME_S >= MAX_UTTERANCE_S
            if silence_s >= END_SILENCE_S or too_long:
                in_speech, preroll, hot = False, [], 0
                audio = b"".join(voiced)
                voiced = []
                if not self.awake:
                    threading.Thread(target=self.check_wake, args=(audio,), daemon=True).start()
                    continue
                log(f"utterance end after {len(audio) / 2 / RATE:.1f}s")
                threading.Thread(target=self.transcribe, args=(audio, barged), daemon=True).start()

        rec.kill()

    def transcribe(self, audio, barged, strip_wake_phrase=False):
        try:
            text = self._transcribe(audio, barged, strip_wake_phrase)
        except Exception as err:  # every `hearing` must end in an utterance or a drop
            log("transcribe error:", repr(err))
            self.emit("dropped", reason="error")
            text = ""
        if barged and text:
            self.flush(self.gen + 1, "barge-in")
            self.emit("bargein", gen=self.gen)
        elif barged:
            self.player.resumed.set()
            log("resumed: not a barge-in")
            if self.speaking:
                self.emit("speaking")
        if text:
            self.last_activity = time.monotonic()
            self.emit("utterance", text=text, barged=barged)

    def _transcribe(self, audio, barged, strip_wake_phrase):
        """Returns the transcript to deliver, or "" after emitting why it was dropped."""
        speech_s = len(audio) / 2 / RATE
        if speech_s < 0.4:
            self.emit("dropped", reason="short")
            return ""
        x = self.np.frombuffer(audio, dtype=self.np.int16).astype(self.np.float32) / 32768.0
        t = time.monotonic()
        with self.stt_lock:
            segs = self.stt.transcribe(x, language="en")
        text = clean_transcript(" ".join(seg.text.strip() for seg in segs))
        log(f"stt {time.monotonic() - t:.2f}s for {speech_s:.1f}s audio: {text!r}")
        is_wake, rest = strip_wake(text)
        # Awake, "hey Claude, ..." loses its greeting; a sentence that merely starts
        # with "Cloud ..." keeps every word.
        greeted = re.match(r"^\s*(hey|hi|okay|ok|yo)\b", text, re.IGNORECASE)
        if is_wake and (strip_wake_phrase or greeted):
            text = clean_transcript(rest)
        now = time.monotonic()
        recent = [s for at, s in self.recent_spoken if now - at < ECHO_WINDOW_S]
        if text and is_echo(text, recent, during_speech=barged):
            log("dropped as echo of our own speech")
            self.emit("dropped", reason="echo")
            return ""
        if not text:
            self.emit("dropped", reason="empty")
        return text

    # lifecycle ----------------------------------------------------------

    def watchdog_loop(self):
        while not self.stopping.is_set():
            time.sleep(1)
            if time.monotonic() - self.last_poll > WATCHDOG_S:
                log("no poll for", WATCHDOG_S, "s; exiting")
                self.stopping.set()
            quiet = time.monotonic() - self.last_activity
            if self.awake and not self.speaking and self.busy == 0 and quiet > SLEEP_AFTER_S > 0:
                self.sleep(f"{quiet:.0f}s quiet")


class UnixHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True

    def server_bind(self):
        self.socket.bind(self.server_address)
        self.server_name, self.server_port = "voice", 0


def make_handler(d: Daemon, server_ref):
    class Handler(BaseHTTPRequestHandler):
        def address_string(self):
            return "unix"

        def log_message(self, *args):
            pass

        def reply(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except BrokenPipeError:  # the health check hangs up once it has read enough
                pass

        def do_GET(self):
            if self.path == "/events":
                self.reply({"events": d.drain(), "speaking": d.speaking,
                            "listening": d.listening, "gen": d.gen, "awake": d.awake})
            elif self.path == "/health":
                self.reply({"ok": True, "pid": os.getpid(), "version": VERSION})
            elif self.path == "/config":
                self.reply({"voice": d.tts.voice.name, "speed": d.tts.speed})
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/say":
                d.say(str(body.get("text", "")), int(body.get("gen", 0)))
            elif self.path == "/config":
                try:
                    d.tts.configure(body.get("voice"), body.get("speed"))
                except (ValueError, OSError) as err:
                    self.reply({"error": str(err)}, 400)
                    return
            elif self.path == "/flush":
                d.flush(int(body.get("gen", d.gen + 1)), str(body.get("why", "mod")))
            elif self.path == "/wake":
                d.wake(str(body.get("why", "mod")))
            elif self.path == "/sleep":
                d.sleep(str(body.get("why", "mod")))
            elif self.path == "/log":
                log("mod:", str(body.get("line", ""))[:300])
            elif self.path == "/listen":
                d.listening = bool(body.get("on", True))
            elif self.path == "/shutdown":
                d.stopping.set()
            else:
                self.send_error(404)
                return
            self.reply({"ok": True, "gen": d.gen})

    return Handler


def request(method, path) -> bytes:
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(1)
        s.connect(str(SOCK))
        s.sendall(f"{method} {path} HTTP/1.0\r\nContent-Length: 0\r\n\r\n".encode())
        return s.recv(4096)


def running_version():
    """The live daemon's version ("" for one too old to say), or None if none is live."""
    try:
        reply = request("GET", "/health")
    except OSError:
        return None
    if b'"ok": true' not in reply:
        return None
    m = re.search(rb'"version": "([^"]*)"', reply)
    return m.group(1).decode() if m else ""


def alive() -> bool:
    return running_version() is not None


def start_detached() -> int:
    version = running_version()
    if version == VERSION:
        print(json.dumps({"status": "running", "socket": str(SOCK)}))
        return 0
    if version is not None:  # an older daemon from before an update: replace it
        try:
            request("POST", "/shutdown")
        except OSError:
            pass
        for _ in range(50):
            time.sleep(0.1)
            if not alive():
                break
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "ab") as logf:
        subprocess.Popen(
            [sys.executable, __file__], stdin=subprocess.DEVNULL, stdout=logf, stderr=logf,
            start_new_session=True,
        )
    for _ in range(250):  # Kokoro takes a few seconds to load and warm up
        time.sleep(0.1)
        if alive():
            print(json.dumps({"status": "started", "socket": str(SOCK)}))
            return 0
    print(f"daemon did not come up; see {LOG}", file=sys.stderr)
    return 1


def main():
    if "--start" in sys.argv:
        sys.exit(start_detached())
    if "--voices" in sys.argv:
        print(json.dumps(list_voices()))
        sys.exit(0)
    if "--check-voice" in sys.argv:
        import numpy as np
        try:
            resolve_voice(sys.argv[sys.argv.index("--check-voice") + 1], np)
        except (ValueError, OSError, IndexError) as err:
            print(err, file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    if alive():
        log("another daemon owns", SOCK)
        sys.exit(0)
    SOCK.unlink(missing_ok=True)

    hold_power()
    d = Daemon()
    server = UnixHTTPServer(str(SOCK), None)
    server.RequestHandlerClass = make_handler(d, server)
    for target in (d.listen_loop, d.synth_loop, d.play_loop, d.watchdog_loop):
        threading.Thread(target=target, daemon=True).start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log("ready on", SOCK)

    d.stopping.wait()
    d.stop_player()
    release_power()
    server.shutdown()
    SOCK.unlink(missing_ok=True)
    log("stopped")
    os._exit(0)  # skip interpreter teardown; onnxruntime threads abort noisily during it


if __name__ == "__main__":
    main()
