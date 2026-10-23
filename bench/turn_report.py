"""How a voice session felt, turn by turn, from the daemon log.

    python bench/turn_report.py [~/.local/share/voice-mode/voiced.log] [--since HH:MM]

For each spoken turn (you stop talking -> Claude's turn completes):
  to text   you stop -> your words are text
  1st sound you stop -> anything plays (a filler counts)
  1st reply you stop -> the first sentence of the actual answer plays
  dead air  silences over GAP_S while the turn ran, and how many fell in tool runs

Needs millisecond timestamps (logs from 0.2.4 on); older lines are skipped.
"""
import re
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

GAP_S = 1.5
LINE = re.compile(r"^(\d\d:\d\d:\d\d\.\d{3}) (.*)$")


def parse(path, since=None):
    events, day, last = [], datetime(2000, 1, 1), None
    for raw in Path(path).read_text(errors="replace").splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        t = datetime.combine(day.date(), datetime.strptime(m[1], "%H:%M:%S.%f").time())
        if last and t < last - timedelta(hours=1):  # past midnight
            day += timedelta(days=1)
            t += timedelta(days=1)
        last = t
        if since and m[1] < since:
            continue
        events.append((t.timestamp(), m[2]))
    return events


def turns(events):
    """Yields one dict per spoken turn."""
    cur, speech_end, text_at = None, None, None
    for t, msg in events:
        if msg.startswith("utterance end"):
            speech_end = t
        elif msg.startswith("stt ") and "audio:" in msg:
            text_at = t
        elif msg.startswith("mod: submit:"):
            cur = {"end": speech_end or t, "text_at": text_at or t, "said": msg.split(":", 2)[2].strip()[:50],
                   "plays": [], "tools": [], "reply": None, "fillers": 0}
        elif cur is None:
            continue
        elif msg.startswith("play "):
            m = re.match(r"play ([\d.]+)s from \+([\d.]+)s", msg)
            start = t + float(m[2])
            cur["plays"].append((start, start + float(m[1])))
            cur["fillers"] += "(filler)" in msg
        elif msg.startswith("first audio for gen") and cur["reply"] is None:
            cur["reply"] = t
        elif msg.startswith("mod: tool start"):
            cur["tools"].append([t, None])
        elif msg.startswith("mod: tool end") and cur["tools"] and cur["tools"][-1][1] is None:
            cur["tools"][-1][1] = t
        elif msg.startswith("mod: turn complete"):
            cur["done"] = t
            yield cur
            cur = None


def measure(turn):
    plays = sorted(turn["plays"])
    first_sound = plays[0][0] if plays else None
    # Silence: from you stopping to the end of the turn, minus what was playing.
    gaps, cursor = [], turn["end"]
    for s, e in plays:
        if s - cursor > GAP_S:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if turn["done"] - cursor > GAP_S:
        gaps.append((cursor, turn["done"]))
    in_tools = sum(1 for g0, g1 in gaps
                   if any(t0 < g1 and (t1 or turn["done"]) > g0 for t0, t1 in turn["tools"]))
    return {
        "said": turn["said"],
        "to_text": turn["text_at"] - turn["end"],
        "first_sound": first_sound - turn["end"] if first_sound else None,
        "first_reply": turn["reply"] - turn["end"] if turn["reply"] else None,
        "dead_air": sum(g1 - g0 for g0, g1 in gaps),
        "gaps": len(gaps),
        "in_tools": in_tools,
        "fillers": turn["fillers"],
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else None
    if since in args:
        args.remove(since)
    path = args[0] if args else Path.home() / ".local/share/voice-mode/voiced.log"
    rows = [measure(t) for t in turns(parse(path, since))]
    if not rows:
        print("no spoken turns with millisecond timestamps yet")
        return
    fmt = lambda x: f"{x:6.2f}s" if x is not None else "     -"
    print(f"{'you said':52} {'to text':>7} {'1st snd':>7} {'1st rep':>7} {'dead air':>9} {'gaps (tools)':>12} {'fillers':>7}")
    for r in rows:
        print(f"{r['said']:52} {fmt(r['to_text'])} {fmt(r['first_sound'])} {fmt(r['first_reply'])} "
              f"{r['dead_air']:8.1f}s {r['gaps']:5} ({r['in_tools']}) {r['fillers']:>9}")
    med = lambda k: statistics.median(r[k] for r in rows if r[k] is not None)
    print(f"\n{len(rows)} turns. Median: to text {med('to_text'):.2f}s, first sound {med('first_sound'):.2f}s, "
          f"first reply {med('first_reply'):.2f}s, dead air per turn {med('dead_air'):.1f}s, "
          f"fillers {sum(r['fillers'] for r in rows)}")


if __name__ == "__main__":
    main()
