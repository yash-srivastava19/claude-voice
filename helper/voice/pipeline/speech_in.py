"""Speech in: mic → VAD → turn detection → speech to text → filters → "utterance".

Every "hearing" event ends in exactly one "utterance" or "dropped" (with a reason:
short, empty, echo, backchannel, mic, error), so the harness never waits forever.

Speech that starts while Claude talks pauses the reply (a possible barge-in). The
transcript then decides: real words flush the reply, a listening noise ("uh-huh")
or our own echo resumes it.
"""

import re
import threading
import time

from voice.audio.turn.silence import STARTED, SilenceTurns
from voice.config import ECHO_TAIL_S, PREROLL_FRAMES, WAKE_MAX_SEGMENT_S, WAKE_WINDOW_S
from voice.frames import FRAME_S, MIC_BACK, MIC_LOST, RATE, seconds, to_float
from voice.log import log
from voice.processors.filters import clean_transcript, heard_wake, is_backchannel, is_echo, strip_wake

TAIL_FRAMES = int(ECHO_TAIL_S / FRAME_S) + 1


class SpeechIn:
    def __init__(self, *, stt, vad, mic, out, session, echo, wake_stt=None):
        """stt, wake_stt: STTService. vad: SileroVAD. mic: Mic. out: SpeechOut.
        session: the VoiceTask (emit, touch, awake, listening, wake).
        echo: "aec", "gate" or "open" (see config.ECHO_MODES)."""
        self.stt, self.wake_stt, self.vad, self.mic = stt, wake_stt, vad, mic
        self.out, self.session, self.echo = out, session, echo
        self.turns = SilenceTurns(TAIL_FRAMES)
        self.barged = False  # the open utterance started while Claude was talking

    def threads(self):
        return (self.listen_loop,)

    # the loop: one mic frame at a time ------------------------------------

    def listen_loop(self):
        s = self.session
        for frame in self.mic.frames(s.stopping):
            if frame == MIC_LOST:
                self._mic_lost()
                continue
            if frame == MIC_BACK:
                s.emit("mic", ok=True)
                continue
            prob = self.vad.prob(frame)  # every frame, so the VAD's state follows the stream
            if not s.listening:
                self.turns.drop()
                self.out.resume()  # a paused reply must not stay paused with the mic off
            elif self._gated():
                self._hold(frame)
            else:
                self._listen(frame, prob)

    def _gated(self):
        """Without echo cancellation, the mic is ignored while Claude speaks and just after."""
        return self.echo == "gate" and (self.out.speaking or time.monotonic() < self._tail_end())

    def _tail_end(self):
        return self.out.sound_end() + ECHO_TAIL_S

    def _hold(self, frame):
        # Speech already under way is kept and sent on. The tail after playback is still
        # recorded, so a reply that starts the moment Claude stops keeps its first word.
        cut = self.turns.hold(frame, self.out.speaking)
        if cut and self.session.awake:
            log(f"utterance cut short by playback after {seconds(cut.audio):.1f}s")
            self._transcribe_async(cut.audio, False, cut.started)

    def _listen(self, frame, prob):
        after_tail = self.echo == "gate" and time.monotonic() < self._tail_end() + PREROLL_FRAMES * FRAME_S
        step = self.turns.feed(frame, prob, self.out.speaking, after_tail)
        if step == STARTED:
            self._started()
        elif step is not None:
            self._ended(step)

    def _started(self):
        self.barged = False
        if not self.session.awake:
            return  # asleep: no barge-in, no events, only a wake check at the end
        if self.out.speaking:
            # Pause rather than stop: if the transcript turns out to be our own echo or
            # noise, the reply carries on where it left off.
            self.barged = True
            self.out.pause()
        log("hearing" + (" (paused for a possible barge-in)" if self.barged else ""))
        self.session.emit("hearing")

    def _ended(self, ended):
        if not self.session.awake:
            threading.Thread(target=self.check_wake, args=(ended.audio,), daemon=True).start()
            return
        log(f"utterance end after {seconds(ended.audio):.1f}s")
        self._transcribe_async(ended.audio, self.barged, ended.started)

    def _mic_lost(self):
        self.session.emit("mic", ok=False)
        if self.turns.in_speech and self.session.awake:
            self.session.emit("dropped", reason="mic")
        self.vad.reset()
        self.turns.reset()

    # transcripts -----------------------------------------------------------

    def _transcribe_async(self, audio, barged, started):
        threading.Thread(target=self.transcribe, daemon=True,
                         args=(audio, barged, started, time.monotonic())).start()

    def transcribe(self, audio, barged, started=None, ended=None, strip_wake_phrase=False):
        ended = ended if ended is not None else time.monotonic()
        started = started if started is not None else ended - seconds(audio)
        try:
            text = self._transcript(audio, started, ended, strip_wake_phrase)
        except Exception as err:  # every `hearing` must end in an utterance or a drop
            log("transcribe error:", repr(err))
            self.session.emit("dropped", reason="error")
            text = ""
        if barged and text and is_backchannel(text):
            log(f"backchannel while speaking, not a barge-in: {text!r}")
            self.session.emit("dropped", reason="backchannel")
            text = ""
        if barged:
            self._settle_barge_in(text)
        if text:
            self.session.touch()
            self.session.emit("utterance", text=text, barged=barged)

    def _settle_barge_in(self, text):
        if text:
            self.out.flush(self.out.gen + 1, "barge-in")
            self.session.emit("bargein", gen=self.out.gen)
            return
        self.out.resume()
        log("resumed: not a barge-in")
        if self.out.speaking:
            self.session.emit("speaking")

    def _transcript(self, audio, started, ended, strip_wake_phrase):
        """The transcript to deliver, or "" after emitting why it was dropped."""
        speech_s = seconds(audio)
        if speech_s < 0.4:
            self.session.emit("dropped", reason="short")
            return ""
        t = time.monotonic()
        text = clean_transcript(self.stt.transcribe(to_float(audio)))
        log(f"stt {time.monotonic() - t:.2f}s for {speech_s:.1f}s audio: {text!r}")
        is_wake, rest = strip_wake(text)
        # Awake, "hey Claude, ..." loses its greeting; a sentence that merely starts
        # with "Cloud ..." keeps every word.
        greeted = re.match(r"^\s*(hey|hi|okay|ok|yo)\b", text, re.IGNORECASE)
        if is_wake and (strip_wake_phrase or greeted):
            text = clean_transcript(rest)
        if text and is_echo(text, self.out.playing_between(started, ended, ECHO_TAIL_S)):
            log("dropped as echo of our own speech")
            self.session.emit("dropped", reason="echo")
            return ""
        if not text:
            self.session.emit("dropped", reason="empty")
        return text

    # asleep ------------------------------------------------------------------

    def check_wake(self, audio):
        """While asleep: wakes on a segment that opens with the wake phrase."""
        speech_s = seconds(audio)
        if speech_s < 0.4 or speech_s > WAKE_MAX_SEGMENT_S:
            return
        t = time.monotonic()
        try:
            heard = self.wake_stt.transcribe(to_float(audio[: int(WAKE_WINDOW_S * RATE) * 2]))
        except Exception as err:
            log("wake check error:", repr(err))
            return
        is_wake = heard_wake(heard)
        log(f"wake check {time.monotonic() - t:.2f}s: {heard!r}{' -> wake' if is_wake else ''}")
        if not is_wake:
            return
        self.session.wake("wake phrase")
        # "Hey Claude, what does this do?" carries a request: transcribe it properly.
        if speech_s > WAKE_WINDOW_S - 0.5:
            self.transcribe(audio, False, strip_wake_phrase=True)
