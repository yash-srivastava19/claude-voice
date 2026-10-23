"""The numbers a voice session feels, measured through the real daemon (fake mic, null
sink, Linux): time from /say to the first sound, the silence between the first two
pieces of a reply, and how long your words take to become text while a long reply is
being synthesized.

    $BENCH/venv/bin/python bench/live.py [daemon.py ...]     (default: this checkout)

Each daemon runs ROUNDS times, alternating. Env VOICE_* passes through, so
VOICE_TTS_NICE=0 measures without the priority change.
"""
import os, re, statistics, sys, tempfile, time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import test_pipeline as tp  # the golden test's daemon harness

ROUNDS = int(os.environ.get("ROUNDS", 3))
OPENERS = ["Sure, the basics are simple, and you can do most of it without thinking about it.",
           "Found it, your website is a Jekyll site hosted on GitHub Pages.",
           "Okay, so the slow part is your laptop, not voice mode."]
LONG = " ".join(["The first check is easy, if I am talking the mic is paused."] * 6)


def stamp(line):
    clock = datetime.strptime(line[:12], "%H:%M:%S.%f").time()
    return datetime.combine(date.today(), clock).timestamp()


def plays(lines):
    """(start, end) of each piece played, from the daemon's play lines."""
    out = []
    for l in lines:
        m = re.search(r" play ([\d.]+)s from \+([\d.]+)s", l)
        if m:
            start = stamp(l) + float(m[2])
            out.append((start, start + float(m[1])))
    return out


def quiet(d):
    end = time.monotonic() + 30
    while d.call("GET", "/events")["speaking"] and time.monotonic() < end:
        time.sleep(0.2)


def one_run(daemon_py, label):
    os.environ["VOICE_DAEMON"] = str(daemon_py)
    tp.DAEMON = Path(daemon_py)
    d = tp.Daemon(Path(tempfile.mkdtemp()))
    first, gaps, stt = [], [], []
    try:
        for gen, text in enumerate(OPENERS):  # first sound after /say
            d.call("POST", "/flush", {"gen": gen * 10})
            mark = len(d.log.read_text().splitlines())
            t = time.time()
            d.call("POST", "/say", {"text": text, "gen": gen * 10})
            d.wait_for(lambda e: "idle" in tp.kinds(e))
            pieces = plays(d.log.read_text().splitlines()[mark:])
            if pieces:
                first.append(pieces[0][0] - t)
            if len(pieces) > 1:
                gaps.append(max(0.0, pieces[1][0] - pieces[0][1]))
        for i in range(4):  # your words while a long reply is synthesized
            gen = 100 + i
            d.call("POST", "/say", {"text": LONG, "gen": gen})
            d.wait_for(lambda e: "speaking" in tp.kinds(e))
            mark = len(d.log.read_text().splitlines())
            d.speak_into_mic("interrupt")
            d.wait_for(lambda e: "utterance" in tp.kinds(e) or "dropped" in tp.kinds(e))
            for l in d.log.read_text().splitlines()[mark:]:
                m = re.search(r" stt ([\d.]+)s for", l)
                if m:
                    stt.append(float(m[1]))
            d.call("POST", "/flush", {"gen": gen + 1})
            quiet(d)
        pid = d.call("GET", "/health")["pid"]
        nices = sorted({int(Path(f"/proc/{pid}/task/{t}/stat").read_text().rsplit(")", 1)[1].split()[16])
                        for t in os.listdir(f"/proc/{pid}/task")})
    finally:
        d.close()
    return first, gaps, stt, nices


def main():
    tp.clips()
    daemons = sys.argv[1:] or [str(tp.ROOT / "helper/voiced.py")]
    labels = [f"{Path(p).parent.parent.name}/{Path(p).name}" for p in daemons]
    res = {l: {"first": [], "gaps": [], "stt": [], "nices": None} for l in labels}
    for r in range(ROUNDS):
        for p, l in (list(zip(daemons, labels)) if r % 2 == 0 else list(zip(daemons, labels))[::-1]):
            first, gaps, stt, nices = one_run(p, l)
            res[l]["first"] += first
            res[l]["gaps"] += gaps
            res[l]["stt"] += stt
            res[l]["nices"] = nices
            print(f"round {r + 1} {l}: first {[round(x, 2) for x in first]} stt {stt}", file=sys.stderr)
    print(f"VOICE_TTS_NICE={os.environ.get('VOICE_TTS_NICE', 'default')}, {ROUNDS} rounds, load {os.getloadavg()[0]:.1f}")
    for l, r in res.items():
        f, g, s = sorted(r["first"]), sorted(r["gaps"]) or [0.0], sorted(r["stt"])
        print(f"{l:40} first sound median {statistics.median(f):.2f}s (range {f[0]:.2f}-{f[-1]:.2f}, n={len(f)})  "
              f"gap after first piece median {statistics.median(g):.2f}s (max {g[-1]:.2f}, n={len(r['gaps'])})  "
              f"stt while speaking median {statistics.median(s):.2f}s (range {s[0]:.2f}-{s[-1]:.2f}, n={len(s)})  "
              f"thread nice values {r['nices']}")


if __name__ == "__main__":
    main()
