"""Speech to text and text to speech running at the same time, as in a barge-in.

    $BENCH/venv/bin/python bench/contention.py STT_ENGINE TTS_ENGINE [stt_threads tts_threads]

Text to speech synthesizes reply sentences in a loop in one process while speech
to text transcribes the short (<=5 s) test clips in another. Prints each side's
latency alone and together.
"""
import json, os, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import stt_eval, tts_eval

BENCH = stt_eval.BENCH


NICE = {"stt": 5, "tts": 5}  # equal unless TTS_NICE says otherwise


def stt_side(name, threads, seconds):
    import soundfile as sf
    os.nice(NICE["stt"])
    stt_eval.THREADS = threads
    run = stt_eval.ENGINES[name]()
    items = [i for i in json.loads((stt_eval.SET / "manifest.json").read_text()) if i["dur"] <= 5][:40]
    clips = [sf.read(stt_eval.SET / i["file"], dtype="float32")[0] for i in items]
    run(clips[0])
    ts, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        for x in clips:
            t = time.perf_counter(); run(x); ts.append(time.perf_counter() - t)
            if time.monotonic() >= end:
                break
    print(json.dumps(ts))


def tts_side(name, threads, seconds):
    os.nice(int(os.environ.get("TTS_NICE", NICE["tts"])))
    e = tts_eval.ENGINES[name]
    run = e() if threads is None else _with_threads(name, threads)
    run(tts_eval.FIRST)
    rtf, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        for text in tts_eval.REPLY:
            t = time.perf_counter(); pcm, rate = run(text); rtf.append((time.perf_counter() - t) / (len(pcm) / rate))
            if time.monotonic() >= end:
                break
    print(json.dumps(rtf))


def _with_threads(name, threads):
    if name.startswith("piper"):
        return tts_eval.piper(name.split()[0].replace("piper-", "vits-piper-en_US-"), threads)
    model = "kokoro-v1.0.int8.onnx" if "int8" in name else "kokoro-v1.0.onnx"
    return tts_eval.kokoro(model, threads)


def spawn(side, name, threads, seconds):
    return subprocess.Popen([sys.executable, __file__, "--" + side, name, str(threads), str(seconds)],
                            stdout=subprocess.PIPE, text=True)


def main():
    if sys.argv[1] in ("--stt", "--tts"):
        th = None if sys.argv[3] == "None" else int(sys.argv[3])
        return (stt_side if sys.argv[1] == "--stt" else tts_side)(sys.argv[2], th, float(sys.argv[4]))
    stt, tts = sys.argv[1], sys.argv[2]
    st = int(sys.argv[3]) if len(sys.argv) > 3 else stt_eval.THREADS
    tt = (None if sys.argv[4] == "None" else int(sys.argv[4])) if len(sys.argv) > 4 else None
    secs = 40
    q = lambda xs: (statistics.median(xs), sorted(xs)[int(0.9 * len(xs))])
    alone_s = json.loads(spawn("stt", stt, st, secs).communicate()[0])
    alone_t = json.loads(spawn("tts", tts, tt, secs).communicate()[0])
    a, b = spawn("stt", stt, st, secs), spawn("tts", tts, tt, secs)
    both_s, both_t = json.loads(a.communicate()[0]), json.loads(b.communicate()[0])
    print(f"{stt} (t={st}) + {tts} (t={tt or 'default'}), tts nice {os.environ.get('TTS_NICE', NICE['tts'])}")
    print(f"  stt ≤5s clip  alone med/p90 {q(alone_s)[0]:.2f}/{q(alone_s)[1]:.2f}s   together {q(both_s)[0]:.2f}/{q(both_s)[1]:.2f}s")
    print(f"  tts rtf       alone med/p90 {q(alone_t)[0]:.2f}/{q(alone_t)[1]:.2f}    together {q(both_t)[0]:.2f}/{q(both_t)[1]:.2f}")


if __name__ == "__main__":
    main()
