"""The voice benchmark: speech to text, text to speech, and both at once, through the
plugin's own engines, on whatever machine runs it.

    ~/.local/share/voice-mode/venv/bin/python bench/run.py [stt|tts|contention|all]
        [--provider cpu|cuda|coreml] [--quick] [--rounds N]

Needs the plugin installed (sh install.sh) plus soundfile, pyarrow and jiwer. Each
config runs in its own low-priority process; rounds alternate the config order so drift
favours none. Results land in bench/results/<machine>-<suite>.json; report.py compares
them across machines.
"""

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "helper"))
sys.path.insert(0, str(HERE))

from voice.config import MODELS, PARAKEET_MODEL, WHISPER_MODEL  # noqa: E402

RESULTS = HERE / "results"
REPLY = [  # typical spoken reply sentences
    "Funes was using four cores, so voice only got half the processor.",
    "I merged the pull request and updated the plugin on this laptop.",
    "The first check is easy: if I am talking, the mic is paused.",
    "Then it waits half a second, because the speakers keep ringing a little.",
    "That middle box is the part we fixed, so your first word is not cut off.",
    "Want me to fix it now, or look at the numbers first?",
    "Okay, opening the voice log.",
    "Right, found it, transcription took almost four seconds.",
]
OPENERS = [  # real first sentences of spoken replies
    "Sure, the basics are simple, and you can do most of it without thinking about it.",
    "Found it, your website is a Jekyll site hosted on GitHub Pages.",
    "Yes, I can hear you clearly, and voice mode is on.",
    "That sounds like my own voice came back through the mic.",
    "Agreed, this is a known step in speech pipelines called text normalization.",
    "I think you mean idempotent operations, and I'm going with that.",
    "Good, then it worked, and it heard you correctly both times.",
    "Okay, so the slow part is your laptop, not voice mode.",
]


# configs ------------------------------------------------------------------

def stt_configs(provider):
    configs = {"whisper-base": ("whisper", {"model": WHISPER_MODEL, "language": "en"})}
    if Path(PARAKEET_MODEL).is_dir():
        configs["parakeet"] = ("parakeet", {"model": PARAKEET_MODEL, "provider": provider})
    return configs


def tts_configs(provider):
    kokoro = {"provider": provider}
    configs = {"scarlet (kokoro fp32)": ("scarlet", {"kokoro": kokoro})}
    if (MODELS / "kokoro-v1.0.int8.onnx").exists():
        configs["scarlet (kokoro int8)"] = ("scarlet", {"kokoro": {**kokoro, "model": str(MODELS / "kokoro-v1.0.int8.onnx")}})
    for voice in ("ryan", "lessac"):
        if (MODELS / "piper" / f"{voice}-medium").is_dir():
            configs[f"{voice} (piper)"] = (voice, {"piper": {"provider": provider}})
    return configs


# what each config process does -------------------------------------------------

def rss_mb():
    import resource
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 / (1024 if sys.platform == "darwin" else 1)  # bytes on macOS, KB on Linux


def run_stt(engine, opts, clips):
    import soundfile as sf
    from voice.services import stt_service
    stt = stt_service.create(engine, **opts)
    stt.transcribe(sf.read(clips[0]["path"], dtype="float32")[0])  # warm-up
    out = []
    for c in clips:
        x = sf.read(c["path"], dtype="float32")[0]
        t = time.perf_counter()
        text = stt.transcribe(x)
        out.append({"file": c["file"], "t": time.perf_counter() - t, "hyp": text})
    return {"items": out, "rss_mb": rss_mb()}


def run_tts(voice, engine_opts, samples_dir):
    import numpy as np
    import soundfile as sf
    from voice.processors.speakable import speakable
    from voice.services.tts_service import TTSSwitcher
    tts = TTSSwitcher(voice, 1.05, engine_opts=engine_opts)
    tts.synthesize("Warming up the model once.")

    def timed(text):
        t = time.perf_counter()
        rate, pcm = tts.synthesize(speakable(text))
        return time.perf_counter() - t, rate, pcm

    first = [timed(s)[0] for s in OPENERS]
    clause = [timed(re.split(r"(?<=,)\s", s, maxsplit=1)[0])[0] for s in OPENERS]
    reply = []
    samples_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(REPLY):
        dt, rate, pcm = timed(s)
        reply.append({"text": s, "t": dt, "audio_s": len(pcm) / 2 / rate})
        sf.write(samples_dir / f"{i}.wav", np.frombuffer(pcm, np.int16), rate)
    return {"first_sentence": first, "first_clause": clause, "reply": reply, "rss_mb": rss_mb()}


def run_side(side, cfg, seconds, clips):
    """One half of the contention suite: loop for `seconds`, return per-call latencies."""
    import soundfile as sf
    end, out = time.monotonic() + seconds, []
    if side == "stt":
        from voice.services import stt_service
        stt = stt_service.create(cfg[0], **cfg[1])
        audio = [sf.read(c["path"], dtype="float32")[0] for c in clips]
        stt.transcribe(audio[0])
        while time.monotonic() < end:
            for x in audio:
                t = time.perf_counter(); stt.transcribe(x); out.append(time.perf_counter() - t)
    else:
        from voice.services.tts_service import TTSSwitcher
        tts = TTSSwitcher(cfg[0], 1.05, engine_opts=cfg[1])
        tts.synthesize("Warming up the model once.")
        while time.monotonic() < end:
            for s in REPLY:
                t = time.perf_counter(); rate, pcm = tts.synthesize(s)
                out.append((time.perf_counter() - t) / (len(pcm) / 2 / rate))
    return out


def child():
    job = json.loads(sys.argv[2])
    os.nice(job.get("nice", 10))
    if job["kind"] == "stt":
        res = run_stt(job["engine"], job["opts"], job["clips"])
    elif job["kind"] == "tts":
        res = run_tts(job["voice"], job["engine_opts"], Path(job["samples"]))
    else:
        res = run_side(job["side"], job["cfg"], job["seconds"], job["clips"])
    print(json.dumps(res))


def spawn(job):
    return subprocess.Popen([sys.executable, __file__, "--child", json.dumps(job)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def result(proc, label):
    out, err = proc.communicate()
    if proc.returncode:
        print(f"  {label} failed: {err.strip().splitlines()[-1:] or proc.returncode}", file=sys.stderr)
        return None
    return json.loads(out.strip().splitlines()[-1])


# suites -----------------------------------------------------------------------

def suite_stt(args, clips):
    configs = stt_configs(args.provider)
    runs = {k: [] for k in configs}
    for r in range(args.rounds):
        for label in (list(configs) if r % 2 == 0 else list(configs)[::-1]):
            engine, opts = configs[label]
            print(f"  stt round {r + 1}: {label}", file=sys.stderr)
            res = result(spawn({"kind": "stt", "engine": engine, "opts": opts, "clips": clips}), label)
            if res:
                runs[label].append(res)
    return runs


def suite_tts(args, _clips):
    configs = tts_configs(args.provider)
    runs = {k: [] for k in configs}
    for r in range(args.rounds):
        for label in (list(configs) if r % 2 == 0 else list(configs)[::-1]):
            voice, engine_opts = configs[label]
            print(f"  tts round {r + 1}: {label}", file=sys.stderr)
            samples = RESULTS / "samples" / re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
            res = result(spawn({"kind": "tts", "voice": voice, "engine_opts": engine_opts, "samples": str(samples)}), label)
            if res:
                runs[label].append(res)
    return runs


def suite_contention(args, clips):
    """The best speech-to-text engine against each voice: alone, together at equal
    priority, and together with text to speech at lower priority."""
    stt_label = "parakeet" if "parakeet" in stt_configs(args.provider) else "whisper-base"
    stt = stt_configs(args.provider)[stt_label]
    short = [c for c in clips if c["dur"] <= 5][:40]
    secs = 15 if args.quick else 40
    out = {}
    for label, cfg in tts_configs(args.provider).items():
        print(f"  contention: {stt_label} + {label}", file=sys.stderr)
        side = lambda s, c, nice: {"kind": "side", "side": s, "cfg": c, "seconds": secs, "clips": short, "nice": nice}
        row = {"stt_alone": result(spawn(side("stt", stt, 5)), "stt"),
               "tts_alone": result(spawn(side("tts", cfg, 5)), "tts")}
        for name, tts_nice in (("equal", 5), ("tts_lower", 15)):
            a, b = spawn(side("stt", stt, 5)), spawn(side("tts", cfg, tts_nice))
            row[f"stt_{name}"], row[f"tts_{name}"] = result(a, "stt"), result(b, "tts")
        out[f"{stt_label} + {label}"] = row
    return out


# machine ----------------------------------------------------------------------

def machine():
    info = {"host": socket.gethostname(), "os": platform.platform(), "python": platform.python_version(),
            "cores": os.cpu_count(), "load": list(os.getloadavg())}
    try:
        if sys.platform == "darwin":
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            info["ram_gb"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
        else:
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
            info["ram_gb"] = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 2**20
    except Exception:
        info["cpu"] = platform.processor()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True) \
        if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0 else None
    info["gpu"] = gpu.stdout.strip() if gpu and gpu.returncode == 0 else None
    import onnxruntime
    info["onnx_providers"] = onnxruntime.get_available_providers()
    return info


def main():
    if sys.argv[1:2] == ["--child"]:
        return child()
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", nargs="?", default="all", choices=["stt", "tts", "contention", "all"])
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "coreml"])
    ap.add_argument("--quick", action="store_true", help="40 clips, 1 round: a ballpark, not a verdict")
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()
    args.rounds = args.rounds or (1 if args.quick else 2)

    import data
    clips = [{**c, "path": str(data.SET / c["file"])} for c in data.load()]
    if args.quick:
        clips = clips[::max(1, len(clips) // 40)]
    m = machine()
    print(f"{m['cpu']}, {m['cores']} cores, {m['ram_gb']:.0f} GB, gpu {m['gpu']}, load {m['load'][0]:.1f}", file=sys.stderr)
    RESULTS.mkdir(exist_ok=True)
    tag = re.sub(r"[^a-z0-9]+", "-", f"{m['host']}-{args.provider}".lower())
    for suite in (["stt", "tts", "contention"] if args.suite == "all" else [args.suite]):
        runs = {"stt": suite_stt, "tts": suite_tts, "contention": suite_contention}[suite](args, clips)
        path = RESULTS / f"{tag}-{suite}.json"
        path.write_text(json.dumps({"machine": m, "suite": suite, "provider": args.provider, "quick": args.quick,
                                    "rounds": args.rounds, "clips": [c["file"] for c in clips], "runs": runs}))
        print(f"wrote {path}", file=sys.stderr)
    subprocess.run([sys.executable, str(HERE / "report.py"), *sorted(str(p) for p in RESULTS.glob(f"{tag}-*.json"))])


if __name__ == "__main__":
    main()
