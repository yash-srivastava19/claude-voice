"""A stand-in mic for tests: streams 16 kHz mono s16 silence in real time, and plays
any .raw clip dropped into CTL_DIR (oldest first, deleted once queued)."""
import sys, time
from pathlib import Path

ctl = Path(sys.argv[1])
FRAME = 512
silence = bytes(FRAME * 2)
pending = b""
out = sys.stdout.buffer
next_t = time.monotonic()
while True:
    if not pending:
        for f in sorted(ctl.glob("*.raw")):
            pending += f.read_bytes()
            f.unlink()
            break
    chunk, pending = (pending[:FRAME * 2], pending[FRAME * 2:]) if pending else (silence, b"")
    out.write(chunk.ljust(FRAME * 2, b"\0"))
    out.flush()
    next_t += FRAME / 16000
    time.sleep(max(0, next_t - time.monotonic()))
