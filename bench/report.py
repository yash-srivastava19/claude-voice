"""Tables from bench/run.py results, one row per machine and config, so machines compare.

    python bench/report.py bench/results/*.json
"""
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def norm(s):
    s = s.lower().replace("-", " ").replace("okay", "ok")
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s).split())


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.9 * len(xs)))]


def machine_name(m):
    gpu = f", {m['gpu']}" if m.get("gpu") else ""
    return f"{m['host']} ({m['cpu'][:28]}, {m['cores']} cores{gpu})"


def stt(doc):
    import jiwer
    sys.path.insert(0, str(HERE))
    import data
    texts = {c["file"]: c for c in data.load()}
    print(f"\nspeech to text on {machine_name(doc['machine'])}, {doc['rounds']} round(s), provider {doc['provider']}")
    print(f"  {'engine':16} {'accented':>9} {'indian':>7} {'read':>6} {'jargon':>7} {'turn<=5s':>9} {'p90':>6} {'RAM':>7}")
    for label, runs in doc["runs"].items():
        if not runs:
            continue
        items = [i for run in runs for i in run["items"]]
        def wer(prefix):
            sel = [i for i in runs[-1]["items"] if texts[i["file"]]["group"].startswith(prefix)]
            return jiwer.wer([norm(texts[i["file"]]["text"]) or "x" for i in sel], [norm(i["hyp"]) for i in sel]) if sel else float("nan")
        short = [i["t"] for i in items if texts[i["file"]]["dur"] <= 5]
        print(f"  {label:16} {wer('accented'):9.1%} {wer('accented:Indian'):7.1%} {wer('read'):6.1%} {wer('jargon'):7.1%} "
              f"{statistics.median(short):8.2f}s {p90(short):5.2f}s {max(r['rss_mb'] for r in runs):5.0f}MB")


def tts(doc):
    print(f"\ntext to speech on {machine_name(doc['machine'])}, {doc['rounds']} round(s), provider {doc['provider']}")
    print(f"  {'voice':24} {'1st clause':>10} {'1st sentence':>12} {'RTF med':>8} {'RTF max':>8} {'RAM':>7}")
    for label, runs in doc["runs"].items():
        if not runs:
            continue
        clause = [t for r in runs for t in r["first_clause"]]
        first = [t for r in runs for t in r["first_sentence"]]
        rtf = [x["t"] / x["audio_s"] for r in runs for x in r["reply"]]
        print(f"  {label:24} {statistics.median(clause):9.2f}s {statistics.median(first):11.2f}s "
              f"{statistics.median(rtf):8.2f} {max(rtf):8.2f} {max(r['rss_mb'] for r in runs):5.0f}MB")


def contention(doc):
    print(f"\nboth at once on {machine_name(doc['machine'])} (speech to text seconds per turn; text to speech RTF, over 1 stutters)")
    print(f"  {'pair':40} {'stt alone':>9} {'equal':>7} {'tts lower':>9}   {'tts alone':>9} {'equal':>7} {'lower':>7}")
    for label, row in doc["runs"].items():
        med = lambda k: statistics.median(row[k]) if row.get(k) else float("nan")
        print(f"  {label:40} {med('stt_alone'):8.2f}s {med('stt_equal'):6.2f}s {med('stt_tts_lower'):8.2f}s   "
              f"{med('tts_alone'):9.2f} {med('tts_equal'):7.2f} {med('tts_tts_lower'):7.2f}")


def main():
    for path in sys.argv[1:]:
        doc = json.loads(Path(path).read_text())
        {"stt": stt, "tts": tts, "contention": contention}[doc["suite"]](doc)


if __name__ == "__main__":
    main()
