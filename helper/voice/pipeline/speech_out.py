"""Speech out: /say → text to speech → speaker.

Two threads, so a slow voice model costs latency on the first sentence only:
synth_loop turns queued sentences into audio while play_loop plays the previous one.

Every sentence carries a generation number. flush(gen) raises the current one and
drops everything older, which is how a barge-in or a new prompt silences a reply.
"""

import queue
import threading
import time

from voice.frames import seconds
from voice.log import log
from voice.processors.filters import chunk_sentence


class SpeechOut:
    def __init__(self, tts, player, emit, touch, stopping):
        """tts: TTSService. player: Player. emit(kind, **data) reports to the harness;
        touch() marks activity, so the session doesn't fall asleep mid-reply."""
        self.tts, self.player, self.emit, self.touch, self.stopping = tts, player, emit, touch, stopping
        self.gen = 0
        self.say_q = queue.Queue()  # (gen, text) waiting to be synthesized
        self.audio_q = queue.Queue()  # (gen, text, rate, pcm) waiting to be played
        self.pending = 0  # sentences taken off say_q and not yet played or dropped
        self.pending_lock = threading.Lock()
        self.chunked_gen = -1  # only a reply's first sentence is split, so it starts sooner
        self.speaking = False
        self.last_played_gen = -1
        self.played = []  # (start, end, sentence) in monotonic time, for echo detection

    def threads(self):
        return (self.synth_loop, self.play_loop)

    # what the rest of the daemon calls -----------------------------------

    def say(self, text, gen):
        if gen < self.gen:
            log(f"say dropped: gen {gen} < {self.gen}")
            return
        self.touch()
        # Clause-sized pieces each get their own intonation, which sounds choppy; only the
        # first sentence of a reply is split, so the first sound comes sooner.
        chunks = chunk_sentence(text) if gen != self.chunked_gen else [text]
        self.chunked_gen = gen
        for chunk in chunks:
            with self.pending_lock:
                self.pending += 1
            self.say_q.put((gen, chunk))

    def flush(self, gen, why):
        self.gen = max(self.gen, gen)
        dropped = 0
        for q in (self.say_q, self.audio_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                self._done_one()
                dropped += 1
        was_playing = self.player.busy()
        self.stop()
        log(f"flush ({why}) -> gen {self.gen}: dropped {dropped} queued, {'cut' if was_playing else 'no'} playback")

    def stop(self):
        self.player.kill()
        self.player.resumed.set()

    def pause(self):
        """Hold playback where it is, while a possible barge-in is checked."""
        self.player.resumed.clear()

    def resume(self):
        self.player.resumed.set()

    def sound_end(self):
        return self.player.sound_end()

    def playing_between(self, start, end, tail_s):
        """The sentences that were sounding (plus tail_s of echo) during [start, end]."""
        return [text for s, e, text in self.played if s < end and e + tail_s > start]

    def idle(self):
        return self.pending == 0

    # threads -------------------------------------------------------------

    def synth_loop(self):
        while not self.stopping.is_set():
            try:
                gen, text = self.say_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen < self.gen:
                self._done_one()
                continue
            rate, pcm = self._synthesize(text)
            if not pcm or gen < self.gen:
                self._done_one()
                continue
            self.audio_q.put((gen, text, rate, pcm))

    def play_loop(self):
        while not self.stopping.is_set():
            try:
                gen, text, rate, pcm = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                self._maybe_idle()
                continue
            if gen >= self.gen:
                if not self.speaking:
                    self.speaking = True
                    self.emit("speaking")
                if gen != self.last_played_gen:
                    self.last_played_gen = gen
                    log(f"first audio for gen {gen}")
                self._play(gen, text, rate, pcm)
            self._done_one()

    # helpers -------------------------------------------------------------

    def _synthesize(self, text):
        try:
            t = time.monotonic()
            rate, pcm = self.tts.synthesize(text)
            if rate:
                log(f"tts {time.monotonic() - t:.2f}s for {seconds(pcm, rate):.1f}s audio")
            return rate, pcm
        except Exception as err:  # keep the loop alive on a bad sentence
            log("tts error:", repr(err))
            return 0, b""

    def _play(self, gen, text, rate, pcm):
        now = time.monotonic()
        start = max(now, self.player.until)
        entry = [start, start + seconds(pcm, rate), text]
        self.played = [e for e in self.played if now - e[1] < 30] + [entry]
        done = self.player.write(pcm, rate, lambda: gen < self.gen or self.stopping.is_set())
        if not done:
            entry[1] = time.monotonic()
        log(f"queued {seconds(pcm, rate):.2f}s of audio in {time.monotonic() - now:.2f}s"
            + ("" if done else ", cut"))

    def _maybe_idle(self):
        quiet = self.pending == 0 and not self.player.busy() and self.player.resumed.is_set()
        if self.speaking and quiet:
            self.speaking = False
            self.emit("idle")

    def _done_one(self):
        with self.pending_lock:
            self.pending = max(0, self.pending - 1)
