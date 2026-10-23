"""Text-to-speech speed and intelligibility across engines.

    BENCH=~/.cache/claude-voice-bench
    $BENCH/venv/bin/python bench/tts_eval.py [engine ...]

Speed: time to first audio for a typical opening sentence and for just its first
clause, and real-time factor (synthesis time / audio length) over typical reply
sentences; under 1.0 keeps up with playback. Intelligibility: each clip is
transcribed back with Parakeet and scored by word error rate. That catches
mumbled or garbled speech, not how natural the voice sounds; for that, listen to
the samples saved in $BENCH/tts_samples/.
"""
import json, os, re, statistics, subprocess, sys, time
from pathlib import Path

BENCH = Path(os.environ.get("BENCH", Path.home() / ".cache/claude-voice-bench"))
M = BENCH / "models"
VM = Path.home() / ".local/share/voice-mode/models"
SAMPLES = BENCH / "tts_samples"
FIRST = "Sure, so the slow part is transcription, not the model itself."
CLAUSE = "Sure, so"
REPLY = [
    "Funes was using four cores, so voice only got half the processor.",
    "I merged the pull request and updated the plugin on this laptop.",
    "The first check is easy: if I am talking, the mic is paused.",
    "Then it waits half a second, because the speakers keep ringing a little.",
    "That middle box is the part we fixed, so your first word is not cut off.",
    "Want me to fix it now, or look at the numbers first?",
    "Okay, opening the voice log.",
    "Right, found it. Transcription took almost four seconds.",
]


def kokoro(model, threads=None, provider=None, venv_ov=False):
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    prov = [provider] if provider else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(VM / model), sess_options=opts, providers=prov)
    k = Kokoro.from_session(sess, str(VM / "voices-v1.0.bin"))
    def run(text):
        pcm, rate = k.create(text, voice="af_sky", speed=1.05, lang="en-us")
        return pcm, rate
    return run


def piper(d, threads=4):
    import sherpa_onnx as so
    d = M / d
    onnx = next(d.glob("*.onnx"))
    cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
        vits=so.OfflineTtsVitsModelConfig(model=str(onnx), tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")),
        num_threads=threads), max_num_sentences=1)
    tts = so.OfflineTts(cfg)
    def run(text):
        import numpy as np
        a = tts.generate(text, sid=0, speed=1.0)
        return np.array(a.samples, dtype="float32"), a.sample_rate
    return run


ENGINES = {
    "kokoro-fp32 (today)": lambda: kokoro("kokoro-v1.0.onnx"),
    "kokoro-fp32 t=4": lambda: kokoro("kokoro-v1.0.onnx", 4),
    "kokoro-int8": lambda: kokoro("kokoro-v1.0.int8.onnx"),
    "kokoro-int8 t=4": lambda: kokoro("kokoro-v1.0.int8.onnx", 4),
    "kokoro-fp32 openvino-cpu": lambda: kokoro("kokoro-v1.0.onnx", provider="OpenVINOExecutionProvider"),
    "piper-ryan-low": lambda: piper("vits-piper-en_US-ryan-low"),
    "piper-ryan-medium": lambda: piper("vits-piper-en_US-ryan-medium"),
    "piper-ryan-medium-int8": lambda: piper("vits-piper-en_US-ryan-medium-int8"),
    "piper-lessac-medium": lambda: piper("vits-piper-en_US-lessac-medium"),
    "piper-ryan-high": lambda: piper("vits-piper-en_US-ryan-high"),
}
OV = {"kokoro-fp32 openvino-cpu"}  # needs the venv with onnxruntime-openvino


def child(name):
    import resource, soundfile as sf
    os.nice(10)
    run = ENGINES[name]()
    run(FIRST)  # warm-up
    out = {"first": [], "clause": [], "reply": []}
    for _ in range(3):
        t = time.perf_counter(); run(FIRST); out["first"].append(time.perf_counter() - t)
        t = time.perf_counter(); run(CLAUSE); out["clause"].append(time.perf_counter() - t)
    SAMPLES.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    for i, text in enumerate(REPLY):
        t = time.perf_counter(); pcm, rate = run(text); dt = time.perf_counter() - t
        out["reply"].append({"text": text, "t": dt, "audio_s": len(pcm) / rate})
        sf.write(SAMPLES / f"{slug}_{i}.wav", pcm, rate)
    out["slug"] = slug
    out["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(out))


def intelligibility(res):
    """Transcribe each saved sample with Parakeet; word error rate against the text."""
    import numpy as np, soundfile as sf, sherpa_onnx as so, jiwer
    d = M / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
    r = so.OfflineRecognizer.from_transducer(encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=4, model_type="nemo_transducer",
        modeling_unit="bpe")
    n = lambda s: " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ").replace("okay", "ok")).split())
    for name, o in res.items():
        hyps = []
        for i, _ in enumerate(REPLY):
            x, sr = sf.read(SAMPLES / f"{o['slug']}_{i}.wav", dtype="float32")
            if sr != 16000:
                idx = np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr)); x = np.interp(idx, np.arange(len(x)), x).astype("float32")
            s = r.create_stream(); s.accept_waveform(16000, x); r.decode_stream(s); hyps.append(s.result.text)
        o["wer"] = jiwer.wer([n(t) for t in REPLY], [n(h) for h in hyps])


def summarize(res):
    print(f"{'engine':26} {'1st sent':>9} {'1st clause':>10} {'rtf med':>8} {'rtf max':>8} {'garble':>7} {'MB':>5}")
    for name, o in res.items():
        rtf = [r["t"] / r["audio_s"] for r in o["reply"]]
        print(f"{name:26} {statistics.median(o['first']):8.2f}s {statistics.median(o['clause']):9.2f}s "
              f"{statistics.median(rtf):8.2f} {max(rtf):8.2f} {o.get('wer', float('nan')):7.1%} {o['rss_mb']:5.0f}")


def main():
    path = BENCH / "tts_results.json"
    if sys.argv[1:2] == ["--child"]:
        return child(sys.argv[2])
    res = json.loads(path.read_text()) if path.exists() else {}
    if sys.argv[1:2] != ["--report"]:
        for name in sys.argv[1:] or ENGINES:
            py = str(BENCH / ("venv-ov" if name in OV else "venv") / "bin/python")
            p = subprocess.run([py, __file__, "--child", name], capture_output=True, text=True)
            if p.returncode:
                print(name, "failed:", p.stderr.strip().splitlines()[-3:], file=sys.stderr); continue
            res[name] = json.loads(p.stdout.strip().splitlines()[-1])
            path.write_text(json.dumps(res))
        intelligibility(res)
        path.write_text(json.dumps(res))
    summarize(res)


if __name__ == "__main__":
    main()
