"""How often does speech arrive in pieces? From the daemon log: for each utterance end,
the gap until the next "hearing". A gap under the submit debounce means the mod joined
the two pieces into one prompt; without the debounce they would be two prompts.

    python bench/experiments/debounce.py [log]
"""
import re, sys
from datetime import datetime
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else Path.home() / ".local/share/voice-mode/voiced.log"
LINE = re.compile(r"^(\d\d:\d\d:\d\d(?:\.\d{3})?) (.*)$")
ends, gaps = None, []
for raw in Path(path).read_text(errors="replace").splitlines():
    m = LINE.match(raw)
    if not m:
        continue
    t = datetime.strptime(m[1], "%H:%M:%S.%f" if "." in m[1] else "%H:%M:%S").timestamp()
    if m[2].startswith("utterance end"):
        ends = t
    elif m[2].startswith("hearing") and ends is not None:
        gaps.append(t - ends)
        ends = None
    elif m[2].startswith("mod: submit"):
        ends = None  # the turn went in before any more speech
n = sum(1 for line in Path(path).read_text(errors="replace").splitlines() if " utterance end" in line)
print(f"{n} utterances; {len(gaps)} followed by more speech before the prompt went in")
for limit in (0.4, 0.7, 1.0, 1.5, 2.0):
    k = sum(1 for g in gaps if 0 <= g < limit)
    print(f"  next speech within {limit:.1f}s of the end: {k} ({k / n:.1%} of utterances)")
