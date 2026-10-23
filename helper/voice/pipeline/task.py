"""The voice session, end to end:

    mic ─▶ SpeechIn ─▶ "utterance" event ─▶ harness (the Claude Code mod, over the socket)
                                                │
    speaker ◀─ SpeechOut ◀─────── /say ◀────────┘

    SpeechIn   speech_in.py   VAD → turn detection → speech to text → filters
    SpeechOut  speech_out.py  text to speech → paced playback; pause, resume, flush
    VoiceTask  this file      wires them together; events, sleep and wake, watchdog

What runs inside each step (which model, which device) comes in through the
interfaces in services/, audio/ and transports/; voiced.py picks them.
"""

import threading
import time

from voice.config import ECHO_MODE, ECHO_MODES, SLEEP_AFTER_S, WATCHDOG_S
from voice.frames import event
from voice.log import log
from voice.pipeline.speech_in import SpeechIn
from voice.pipeline.speech_out import SpeechOut


def echo_mode(aec):
    if ECHO_MODE and ECHO_MODE not in ECHO_MODES:
        log(f"VOICE_ECHO={ECHO_MODE!r} is not one of {', '.join(ECHO_MODES)}; choosing for you")
    return ECHO_MODE if ECHO_MODE in ECHO_MODES else ("aec" if aec else "gate")


class VoiceTask:
    def __init__(self, *, stt, tts, vad, mic, player, echo, wake_stt=None):
        """wake_stt is None for no auto-sleep."""
        self.stopping = threading.Event()
        self.events, self.events_lock = [], threading.Lock()
        self.last_poll = self.last_activity = time.monotonic()
        self.listening, self.awake = True, True
        self.can_sleep = wake_stt is not None
        self.echo = echo
        self.out = SpeechOut(tts, player, self.emit, self.touch, self.stopping)
        self.into = SpeechIn(stt=stt, vad=vad, mic=mic, out=self.out, session=self, echo=echo, wake_stt=wake_stt)

    def threads(self):
        return (*self.into.threads(), *self.out.threads(), self.watchdog_loop)

    # events for the harness ------------------------------------------------

    def emit(self, kind, **data):
        with self.events_lock:
            self.events.append(event(kind, **data))

    def drain(self):
        self.last_poll = time.monotonic()
        with self.events_lock:
            out, self.events = self.events, []
        return out

    # sleep and wake: after SLEEP_AFTER_S of quiet, only "hey Claude" gets through ---

    def touch(self):
        self.last_activity = time.monotonic()

    def sleep(self, why):
        if self.awake and self.can_sleep:
            self.awake = False
            log(f"asleep ({why})")
            self.emit("sleep", why=why)

    def wake(self, why):
        self.touch()
        if not self.awake:
            self.awake = True
            log(f"awake ({why})")
            self.emit("wake", why=why)

    # lifecycle -------------------------------------------------------------

    def watchdog_loop(self):
        """Exits when the harness stops polling; sleeps after a quiet spell."""
        while not self.stopping.is_set():
            time.sleep(1)
            if time.monotonic() - self.last_poll > WATCHDOG_S:
                log("no poll for", WATCHDOG_S, "s; exiting")
                self.stopping.set()
            quiet = time.monotonic() - self.last_activity
            if self.awake and not self.out.speaking and self.out.idle() and quiet > SLEEP_AFTER_S > 0:
                self.sleep(f"{quiet:.0f}s quiet")
