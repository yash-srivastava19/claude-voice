"""Turn detection by silence: an utterance starts after a few frames of speech and ends
after END_SILENCE_S of quiet. Pure state, no threads or I/O: feed it frames with
their speech probability and it says when a turn starts and ends."""

import time
from dataclasses import dataclass

from voice.config import (BARGE_FRAMES, BARGE_THRESHOLD, END_SILENCE_S, MAX_UTTERANCE_S, PREROLL_FRAMES,
                          START_FRAMES, THRESHOLD)
from voice.frames import FRAME_S


@dataclass
class Ended:
    audio: bytes
    started: float  # monotonic time the utterance began, preroll included


STARTED = "started"


class SilenceTurns:
    def __init__(self, tail_frames: int):
        self.tail_frames = tail_frames  # extra preroll kept just after a gated echo tail
        self.reset()

    def reset(self):
        """Forget everything, as after the mic restarts."""
        self.preroll = []
        self.drop()

    def drop(self):
        """Abandon the utterance under way, keeping the preroll."""
        self.voiced, self.in_speech, self.hot = [], False, 0
        self.silence_s, self.started = 0.0, 0.0

    def hold(self, frame, speaking):
        """A frame heard while the mic is gated. Returns the utterance it cut short, if any."""
        cut = Ended(b"".join(self.voiced), self.started) if self.in_speech else None
        self.drop()
        self.preroll = [] if speaking else (self.preroll + [frame])[-(PREROLL_FRAMES + self.tail_frames):]
        return cut

    def feed(self, frame, prob, speaking, after_tail=False):
        """A frame and its speech probability. Returns STARTED, an Ended, or None.

        speaking: Claude is talking, so starting takes longer, more confident speech.
        after_tail: just after a gated echo tail, whose audio is kept in front of the speech.
        """
        threshold = BARGE_THRESHOLD if speaking else THRESHOLD
        need = BARGE_FRAMES if speaking else START_FRAMES
        self.hot = self.hot + 1 if prob >= threshold else 0

        if not self.in_speech:
            keep = PREROLL_FRAMES + (self.tail_frames if after_tail else 0)
            self.preroll = (self.preroll + [frame])[-keep:]
            if self.hot < need:
                return None
            self.in_speech, self.voiced, self.silence_s = True, list(self.preroll), 0.0
            self.started = time.monotonic() - len(self.preroll) * FRAME_S
            return STARTED

        self.voiced.append(frame)
        self.silence_s = 0.0 if prob >= THRESHOLD else self.silence_s + FRAME_S
        too_long = len(self.voiced) * FRAME_S >= MAX_UTTERANCE_S
        if self.silence_s < END_SILENCE_S and not too_long:
            return None
        ended = Ended(b"".join(self.voiced), self.started)
        self.preroll = []
        self.drop()
        return ended
