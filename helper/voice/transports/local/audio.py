"""The local machine's mic and speaker, driven through each platform's command-line tools.

Mic: frames() yields FRAME-sized PCM frames and, when the recorder dies (the audio
server restarted, a device went away), MIC_LOST once, then retries until frames
flow again and yields MIC_BACK. Callers reset any per-stream state on "lost".

Player: one long-lived playback stream fed in short paced blocks; see its docstring.
"""

import subprocess
import threading
import time

from voice.config import (AEC_SINK, AEC_SOURCE, IS_MAC, MIC_CMD, MIC_SOURCE, PLAY_BLOCK_S,
                          PLAY_LEAD_S, SPEAKER_SINK)
from voice.frames import FRAME, MIC_BACK, MIC_LOST, RATE
from voice.log import log


def pw_node_exists(kind: str, name: str) -> bool:
    """True when PipeWire has a sink or source (kind "sinks" or "sources") with this name."""
    try:
        out = subprocess.run(["pactl", "list", "short", kind],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.split("\t")[1:2] == [name] for line in out.splitlines())


def pick_devices():
    """(mic source, speaker sink, whether they are the echo-cancelled pair); "" is the default."""
    source = MIC_SOURCE or (AEC_SOURCE if pw_node_exists("sources", AEC_SOURCE) else "")
    sink = SPEAKER_SINK or (AEC_SINK if pw_node_exists("sinks", AEC_SINK) else "")
    return source, sink, source == AEC_SOURCE and sink == AEC_SINK and not MIC_CMD


class Mic:
    def __init__(self, source: str):
        if MIC_CMD:
            self.cmd = MIC_CMD.split()
        elif IS_MAC:
            self.cmd = ["rec", "-q", "-t", "raw", "-r", str(RATE), "-e", "signed", "-b", "16", "-c", "1", "-"]
        elif source:
            self.cmd = ["pw-cat", "--record", "--raw", "--target", source,
                        "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"]
        else:
            self.cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"]
        log("mic:", " ".join(self.cmd[:4]), source or "default source")

    def frames(self, stopping: threading.Event):
        rec = self._record()
        lost = 0  # recorder restarts since audio last came in
        try:
            while not stopping.is_set():
                raw = rec.stdout.read(FRAME * 2)
                if len(raw) < FRAME * 2:
                    # The recorder quit. Keep trying, so voice mode comes back when the mic does.
                    rec.kill()
                    rec.wait()
                    if stopping.is_set():
                        return
                    if lost == 0:
                        log("mic stream ended; reconnecting")
                        yield MIC_LOST
                    lost += 1
                    stopping.wait(min(5.0, 0.5 * lost))
                    rec = self._record()
                    continue
                if lost:
                    log(f"mic back after {lost} restart(s)")
                    yield MIC_BACK
                    lost = 0
                yield raw
        finally:
            rec.kill()

    def _record(self):
        return subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


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
        self.cut_at = 0.0  # when kill last stopped sound mid-way
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

    def sound_end(self):
        """When the speaker last went (or will go) quiet."""
        return max(self.until, self.cut_at)

    def kill(self):
        with self.lock:
            proc, self.proc = self.proc, None
        if self.until > time.monotonic():
            self.cut_at = time.monotonic()
        self.until = 0.0
        if proc and proc.poll() is None:
            proc.kill()
