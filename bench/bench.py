"""Latency benchmark for the voice daemon's models, run on the machine itself.

    ~/.local/share/voice-mode/venv/bin/python bench/bench.py [--rounds 2] [--only stt|tts]

Each config runs in its own low-priority process (so models don't compete for
memory, and the bench can't starve the audio server), and rounds interleave the
configs so thermal throttling doesn't favour the first ones. Results are saved
after every config. Test speech is made with Kokoro: fine for latency, not for
accuracy.
"""
import argparse, json, os, platform, statistics, subprocess, sys, time
from pathlib import Path

HOME = Path(os.environ.get("VOICE_HOME", Path.home() / ".local/share/voice-mode"))
MODELS = HOME / "models"
CLIPS = Path(__file__).with_name("clips")
RATE = 16000
TEXTS = {
    "1s": "Run the tests again.",
    "5s": "Can you check why the voice log shows a slow transcription when the laptop is busy?",
    "20s": ("Okay, so there are a few things I want to look at today. First, the speech to text "
            "step feels slow on this laptop. Second, I want the plugin to work with other coding "
            "agents, not just this one. And third, I want real numbers before we change anything."),
}
FIRST_SENTENCE = "Sure, so the slow part is transcription, not the model."
LONG_SENTENCE = ("Transcription takes about two seconds no matter how long you talk, because the "
                 "model always processes a thirty second window, so shorter windows should help.")


def make_clips():
    import numpy as np, wave
    from kokoro_onnx import Kokoro
    CLIPS.mkdir(exist_ok=True)
    k = None
    for name, text in TEXTS.items():
        p = CLIPS / f"{name}.wav"
        if p.exists():
            continue
        k = k or Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
        pcm, rate = k.create(text, voice="am_michael", speed=1.0, lang="en-us")
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * RATE / rate))  # 24 kHz -> 16 kHz
        pcm = np.interp(idx, np.arange(len(pcm)), pcm)
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
            w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())


def load_clip(name):
    import numpy as np, wave
    with wave.open(str(CLIPS / f"{name}.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def wer(ref, hyp):
    import re
    r, h = re.findall(r"[a-z']+", ref.lower()), re.findall(r"[a-z']+", hyp.lower())
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


def run_stt(cfg, reps):
    from pywhispercpp.model import Model
    m = Model(str(MODELS / f"ggml-{cfg['model']}.bin"), n_threads=cfg["threads"], print_progress=False,
              print_realtime=False, redirect_whispercpp_logs_to=None)
    out = {}
    for name in TEXTS:
        x = load_clip(name)
        kw = {"language": "en"}
        if cfg["ctx"]:
            kw["audio_ctx"] = min(1500, int(len(x) / RATE * 50) + 64)
        m.transcribe(x, **kw)  # warm-up
        times, text = [], ""
        for _ in range(reps):
            t = time.perf_counter()
            text = " ".join(s.text for s in m.transcribe(x, **kw))
            times.append(time.perf_counter() - t)
        out[name] = {"audio_s": len(x) / RATE, "times": times, "wer": wer(TEXTS[name], text), "text": text.strip()}
    return out


def run_tts(cfg, reps):
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = cfg["threads"]
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(MODELS / f"kokoro-v1.0{'.int8' if cfg['int8'] else ''}.onnx"),
                                sess_options=opts, providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(sess, str(MODELS / "voices-v1.0.bin"))
    out = {}
    for name, text in [("first", FIRST_SENTENCE), ("long", LONG_SENTENCE)]:
        k.create(text, voice="af_sky", speed=1.05, lang="en-us")  # warm-up
        times, audio_s = [], 0
        for _ in range(reps):
            t = time.perf_counter()
            pcm, rate = k.create(text, voice="af_sky", speed=1.05, lang="en-us")
            times.append(time.perf_counter() - t)
            audio_s = len(pcm) / rate
        out[name] = {"audio_s": audio_s, "times": times}
    return out


def configs(only):
    cpus = os.cpu_count() or 4
    threads = sorted({2, min(4, cpus)})  # more than 4 starved PipeWire on a 2P+8E laptop
    c = []
    if only in (None, "stt"):
        # base.en at min(8, cpus) threads with the full window is what the daemon runs today.
        c += [{"kind": "stt", "model": "base.en", "threads": t, "ctx": ctx}
              for t in sorted({4, min(8, cpus)}) for ctx in (False, True)]
        c += [{"kind": "stt", "model": "tiny.en", "threads": 4, "ctx": ctx} for ctx in (False, True)]
    if only in (None, "tts"):
        c += [{"kind": "tts", "int8": q, "threads": t} for q in (False, True) for t in threads]
    return c


def label(c):
    if c["kind"] == "stt":
        return f"stt {c['model']:8} t={c['threads']} ctx={'short' if c['ctx'] else 'full '}"
    return f"tts {'int8' if c['int8'] else 'fp32'} t={c['threads']}"


def machine():
    info = {"platform": platform.platform(), "cpus": os.cpu_count(), "load": os.getloadavg()}
    try:
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        info["power_profile"] = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True).stdout.strip()
        info["on_ac"] = any(p.read_text().strip() == "1" for p in Path("/sys/class/power_supply").glob("A*/online"))
    except Exception:
        info["cpu"] = info.get("cpu") or platform.processor()
    return info


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        cfg = json.loads(sys.argv[2])
        os.nice(10)
        res = (run_stt if cfg["kind"] == "stt" else run_tts)(cfg, cfg["reps"])
        print(json.dumps(res))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3, help="timed runs per config per round")
    ap.add_argument("--only", choices=["stt", "tts"])
    ap.add_argument("--out", default=str(Path(__file__).with_name("results.json")))
    a = ap.parse_args()
    make_clips()
    cfgs = configs(a.only)
    results = {label(c): {"cfg": c, "runs": []} for c in cfgs}
    for rnd in range(a.rounds):
        for c in (cfgs if rnd % 2 == 0 else cfgs[::-1]):
            p = subprocess.run([sys.executable, __file__, "--child", json.dumps({**c, "reps": a.reps})],
                               capture_output=True, text=True)
            if p.returncode:
                print(label(c), "failed:", p.stderr.strip().splitlines()[-1:], file=sys.stderr)
                continue
            results[label(c)]["runs"].append(json.loads(p.stdout.strip().splitlines()[-1]))
            Path(a.out).write_text(json.dumps({"machine": machine(), "results": results}, indent=1))
            print(f"round {rnd + 1}: {label(c)}", file=sys.stderr)
    report(a.out)


def report(path):
    d = json.loads(Path(path).read_text())
    print(json.dumps(d["machine"]))
    for lab, r in d["results"].items():
        if not r["runs"]:
            continue
        cells = []
        for item in r["runs"][0]:
            ts = sorted(t for run in r["runs"] for t in run[item]["times"])
            med, p90 = statistics.median(ts), ts[min(len(ts) - 1, int(0.9 * len(ts)))]
            extra = f" wer {r['runs'][-1][item]['wer']:.0%}" if "wer" in r["runs"][-1][item] else \
                    f" rtf {med / r['runs'][-1][item]['audio_s']:.2f}"
            cells.append(f"{item} {med:.2f}/{p90:.2f}s{extra}")
        print(f"{lab:32} " + " | ".join(cells))


if __name__ == "__main__":
    main()
