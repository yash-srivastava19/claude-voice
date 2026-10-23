"""Streaming transcription: transcribe each stretch at a pause while the person is still
talking, so only the last stretch is left when they stop. Against transcribing it all at
the end (today). Real speech, Parakeet, this machine.

    $BENCH/venv/bin/python bench/streaming_stt.py
"""
import json, os, statistics, sys, time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report import norm  # same normalisation as the accuracy benchmark
from turn_detect import RATE, FRAME, VAD, speech_probs

BENCH = Path(os.environ.get("BENCH", Path.home() / ".cache/claude-voice-bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "helper"))
from voice.services.parakeet.stt import ParakeetSTT  # the plugin's own adapter


def pauses(x, vad, min_frames):
    """Sample indexes where a pause of min_frames frames starts, with speech on both sides."""
    probs, cuts, i = speech_probs(x, vad), [], 1
    while i < len(probs) - min_frames:
        if probs[i - 1] >= 0.5 and all(p < 0.5 for p in probs[i:i + min_frames]):
            cuts.append((i + min_frames // 2) * FRAME)  # cut in the middle of the pause
            i += min_frames
        i += 1
    return cuts


def longest_pause(x, vad):
    probs = speech_probs(x, vad)
    best, best_len, i = None, 4, 1
    while i < len(probs):
        if probs[i - 1] >= 0.5 and probs[i] < 0.5:
            j = i
            while j < len(probs) and probs[j] < 0.5:
                j += 1
            if j < len(probs) and j - i > best_len:  # speech resumes after it
                best, best_len = (i + (j - i) // 2) * FRAME, j - i
            i = j
        i += 1
    return [best] if best else []


def main():
    import onnxruntime as ort
    vad = ort.InferenceSession(str(VAD))
    stt = ParakeetSTT(model=Path.home() / ".local/share/voice-mode/models/parakeet-tdt-0.6b-v2-int8")
    items = [i for i in json.loads((BENCH / "set/manifest.json").read_text())
             if i["group"].split(":")[0] in ("accented", "read") and i["dur"] >= 4]
    stt.transcribe(np.zeros(RATE, np.float32))
    strategies = {"pause 150ms": 5, "pause 250ms": 8, "pause 400ms": 13, "longest pause": None}
    rows = {k: {"final": [], "hyp": [], "keeps_up": []} for k in ("today", *strategies)}
    refs = []
    for it in items:
        x = sf.read(BENCH / "set" / it["file"], dtype="float32")[0]
        refs.append(norm(it["text"]) or "x")
        t = time.perf_counter(); full = stt.transcribe(x); rows["today"]["final"].append(time.perf_counter() - t)
        rows["today"]["hyp"].append(norm(full))
        for name, frames in strategies.items():
            if frames is None:  # one cut, at the longest pause of 150 ms or more
                cuts = longest_pause(x, vad)
            else:
                cuts = pauses(x, vad, frames)
            bounds = [0, *cuts, len(x)]
            parts, slow = [], False
            for a, b in zip(bounds, bounds[1:]):
                t = time.perf_counter(); parts.append(stt.transcribe(x[a:b])); dt = time.perf_counter() - t
                if b != len(x) and dt > (bounds[bounds.index(b) + 1] - b) / RATE:
                    slow = True  # this stretch wasn't done before the next one ended
            rows[name]["final"].append(dt)  # only the last stretch is left after they stop
            rows[name]["hyp"].append(norm(" ".join(parts)))
            rows[name]["keeps_up"].append(not slow)
    print(f"{len(items)} clips of real speech, 4-15 s")
    print(f"{'strategy':14} {'final text after you stop':>27} {'word errors':>12} {'background keeps up':>20}")
    for name, r in rows.items():
        f = sorted(r["final"])
        keep = f"{sum(r['keeps_up']) / len(r['keeps_up']):.0%}" if r["keeps_up"] else "-"
        print(f"{name:14} {'median %.2fs, p90 %.2fs' % (statistics.median(f), f[int(.9 * len(f))]):>27} "
              f"{jiwer.wer(refs, r['hyp']):11.1%} {keep:>20}")


if __name__ == "__main__":
    main()
