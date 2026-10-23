"""Speech-to-text accuracy and latency across engines, on a fixed test set.

    BENCH=~/.cache/claude-voice-bench
    $BENCH/venv/bin/python bench/stt_eval.py [engine ...]

The test set ($BENCH/set/manifest.json) mixes real accented speech (EdAcc),
read speech (LibriSpeech test-clean) and synthetic coding jargon. Each engine
runs in its own low-priority process; results go to $BENCH/stt_results.json.
"""
import json, os, re, statistics, subprocess, sys, time
from pathlib import Path

BENCH = Path(os.environ.get("BENCH", Path.home() / ".cache/claude-voice-bench"))
SET = BENCH / "set"
M = BENCH / "models"
VM = Path.home() / ".local/share/voice-mode/models"
THREADS = min(8, os.cpu_count() or 4)
# Generic words a coding assistant hears, not taken from the test sentences.
VOCAB = ["Claude", "Claude Code", "harness", "plugin", "terminal", "git", "commit", "repo",
         "PR", "API", "CLI", "Python", "TypeScript", "Mac", "Linux"]
HINT = "Talking to Claude Code about a " + ", ".join(VOCAB[2:]) + "."


def whisper(model, short, hint=False):
    from pywhispercpp.model import Model
    m = Model(str(VM / f"ggml-{model}.bin"), n_threads=THREADS, print_progress=False,
              print_realtime=False, redirect_whispercpp_logs_to=None)
    def run(x):
        kw = {"language": "en"}
        if short:
            kw["audio_ctx"] = min(1500, int(len(x) / 16000 * 50) + 64)
        if hint:
            kw["initial_prompt"] = HINT
        return " ".join(s.text for s in m.transcribe(x, **kw))
    return run


def sherpa(kind, d, hot=False):
    import sherpa_onnx as so
    d = M / d
    if kind == "moonshine":
        r = so.OfflineRecognizer.from_moonshine(preprocessor=str(d / "preprocess.onnx"), encoder=str(d / "encode.int8.onnx"),
            uncached_decoder=str(d / "uncached_decode.int8.onnx"), cached_decoder=str(d / "cached_decode.int8.onnx"),
            tokens=str(d / "tokens.txt"), num_threads=THREADS)
    elif kind == "moonshine2":
        r = so.OfflineRecognizer.from_moonshine_v2(encoder=str(d / "encoder_model.ort"),
            decoder=str(d / "decoder_model_merged.ort"), tokens=str(d / "tokens.txt"), num_threads=THREADS)
    else:
        kw = {}
        if hot:
            hw = BENCH / "hotwords.txt"
            hw.write_text("\n".join(w.upper() for w in VOCAB) + "\n")
            kw = dict(decoding_method="modified_beam_search", hotwords_file=str(hw), hotwords_score=2.0)
        r = so.OfflineRecognizer.from_transducer(encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=THREADS,
            model_type="nemo_transducer", modeling_unit="bpe", **kw)
    def run(x):
        s = r.create_stream(); s.accept_waveform(16000, x); r.decode_stream(s)
        return s.result.text
    return run


ENGINES = {
    "whisper-base-full (today)": lambda: whisper("base.en", False),
    "whisper-base-short": lambda: whisper("base.en", True),
    "whisper-base-short+hint": lambda: whisper("base.en", True, True),
    "whisper-tiny-short": lambda: whisper("tiny.en", True),
    "moonshine-tiny": lambda: sherpa("moonshine", "sherpa-onnx-moonshine-tiny-en-int8"),
    "moonshine-base": lambda: sherpa("moonshine", "sherpa-onnx-moonshine-base-en-int8"),
    "moonshine2-tiny": lambda: sherpa("moonshine2", "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27"),
    "moonshine2-base": lambda: sherpa("moonshine2", "sherpa-onnx-moonshine-base-en-quantized-2026-02-27"),
    "parakeet-v2": lambda: sherpa("transducer", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"),
    "parakeet-v2+hotwords": lambda: sherpa("transducer", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", True),
}


def norm(s):
    s = s.lower().replace("-", " ").replace("okay", "ok")
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return " ".join(s.split())


def child(name):
    import numpy as np, soundfile as sf, resource
    os.nice(10)
    run = ENGINES[name]()
    items = json.loads((SET / "manifest.json").read_text())
    run(sf.read(SET / items[0]["file"], dtype="float32")[0])  # warm-up
    out = []
    for it in items:
        x = sf.read(SET / it["file"], dtype="float32")[0]
        t = time.perf_counter(); hyp = run(x); dt = time.perf_counter() - t
        out.append({**it, "hyp": hyp.strip(), "t": dt})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"items": out, "rss_mb": rss}))


def summarize(res):
    import jiwer
    rows = []
    for name, r in res.items():
        it = r["items"]
        def wer(g):
            sel = [i for i in it if i["group"].startswith(g)]
            return jiwer.wer([norm(i["text"]) or "x" for i in sel], [norm(i["hyp"]) for i in sel])
        short = sorted(i["t"] for i in it if i["dur"] <= 5)
        allt = sorted(i["t"] for i in it)
        rtf = sum(i["t"] for i in it) / sum(i["dur"] for i in it)
        rows.append((name, wer("accented"), wer("accented:Indian"), wer("read"), wer("jargon"),
                     statistics.median(short), allt[int(0.9 * len(allt))], rtf, r["rss_mb"]))
    print(f"{'engine':27} {'accent':>7} {'indian':>7} {'read':>6} {'jargon':>7} {'≤5s med':>8} {'p90':>6} {'rtf':>5} {'MB':>5}")
    for n, a, i, rd, j, s, p, rtf, mb in rows:
        print(f"{n:27} {a:7.1%} {i:7.1%} {rd:6.1%} {j:7.1%} {s:7.2f}s {p:5.2f}s {rtf:5.3f} {mb:5.0f}")


def main():
    if sys.argv[1:2] == ["--child"]:
        return child(sys.argv[2])
    if sys.argv[1:2] == ["--report"]:
        return summarize(json.loads((BENCH / "stt_results.json").read_text()))
    path = BENCH / "stt_results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for name in sys.argv[1:] or ENGINES:
        p = subprocess.run([sys.executable, __file__, "--child", name], capture_output=True, text=True)
        if p.returncode:
            print(name, "failed:", p.stderr.strip().splitlines()[-3:], file=sys.stderr); continue
        res[name] = json.loads(p.stdout.strip().splitlines()[-1])
        path.write_text(json.dumps(res))
        print("done:", name, file=sys.stderr)
    summarize(res)


if __name__ == "__main__":
    main()
