"""Smart Turn v3.2 on real speech: can it tell "finished" from "pausing mid-sentence"?

    $BENCH/venv/bin/python bench/turn_detect.py

Finished: whole clips from the test set (EdAcc conversation, LibriSpeech reading).
Not finished: the same clips cut at a real pause between 25% and 75% of the way in
(Silero VAD finds pauses of 150 ms or more). Prints accuracy on each, the CPU time per
call, and the wait a Smart Turn policy would give, next to today's fixed wait.
"""
import json, os, statistics, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

BENCH = Path(os.environ.get("BENCH", Path.home() / ".cache/claude-voice-bench"))
sys.path.insert(0, str(BENCH / "ref"))
from _whisper_features import compute_whisper_log_mel_features  # vendored from Pipecat

MODEL = BENCH / "models/smart-turn-v3.2-cpu.onnx"
VAD = Path.home() / ".local/share/voice-mode/models/silero_vad.onnx"
RATE, FRAME = 16000, 512


def speech_probs(x, vad):
    state, ctx, sr, probs = np.zeros((2, 1, 128), np.float32), np.zeros((1, 64), np.float32), np.array(RATE), []
    for i in range(0, len(x) - FRAME + 1, FRAME):
        f = x[i:i + FRAME].reshape(1, -1)
        out, state = vad.run(None, {"input": np.concatenate([ctx, f], 1), "state": state, "sr": sr})
        ctx = f[:, -64:]
        probs.append(float(out[0][0]))
    return probs


def inner_pause(x, vad):
    """Sample index of the end of speech before a >=150 ms pause in the middle half, or None."""
    probs = speech_probs(x, vad)
    n, need = len(probs), 5  # 5 frames x 32 ms
    for i in range(int(n * 0.25), int(n * 0.75)):
        if probs[i - 1] >= 0.5 and all(p < 0.5 for p in probs[i:i + need]):
            return i * FRAME
    return None


def last_8s(x):
    n = 8 * RATE
    return x[-n:] if len(x) >= n else np.pad(x, (n - len(x), 0))


def main():
    import onnxruntime as ort
    items = [i for i in json.loads((BENCH / "set/manifest.json").read_text())
             if i["group"].split(":")[0] in ("accented", "read") and i["dur"] >= 3]
    vad = ort.InferenceSession(str(VAD))
    res = {}
    for threads in (1, 2):
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        st = ort.InferenceSession(str(MODEL), sess_options=so)
        def p_done(x):
            """(probability the speaker has finished, seconds the call took)."""
            feats = compute_whisper_log_mel_features(last_8s(x), do_normalize=True)[None]
            t = time.perf_counter()
            p = st.run(None, {"input_features": feats})[0][0].item()
            return p, time.perf_counter() - t
        done, notdone, times = [], [], []
        for it in items:
            x = sf.read(BENCH / "set" / it["file"], dtype="float32")[0]
            # Pipecat checks 200 ms into the silence after speech; clips end at the last word.
            p, dt = p_done(np.concatenate([x, np.zeros(int(0.2 * RATE), np.float32)]))
            done.append((p, it["group"]))
            times.append(dt)
            cut = inner_pause(x, vad)
            if cut:  # what the mic has 200 ms into the pause, when the check would run
                p, dt = p_done(x[:cut + int(0.2 * RATE)])
                notdone.append((p, it["group"]))
                times.append(dt)
        res[threads] = (done, notdone, times)
    done, notdone, times = res[1]
    print(f"{len(done)} finished clips, {len(notdone)} cut at a mid-sentence pause")
    for th, (_, _, ts) in res.items():
        print(f"  CPU per call, {th} thread(s): median {statistics.median(ts) * 1000:.0f} ms, p90 {sorted(ts)[int(.9 * len(ts))] * 1000:.0f} ms")
    for thr in (0.3, 0.5, 0.7):
        tp = sum(p >= thr for p, _ in done) / len(done)
        tn = sum(p < thr for p, _ in notdone) / len(notdone)
        print(f"  threshold {thr}: finished called finished {tp:.0%}; mid-sentence pause called not finished {tn:.0%}")
    for g in ("accented", "read"):
        d = [p for p, grp in done if grp.startswith(g)]
        nd = [p for p, grp in notdone if grp.startswith(g)]
        print(f"  {g:9} at 0.5: finished {sum(p >= .5 for p in d) / len(d):.0%} (n={len(d)}), "
              f"pause {sum(p < .5 for p in nd) / max(1, len(nd)):.0%} (n={len(nd)})")
    # The policy: after 200 ms of quiet ask Smart Turn; finished -> submit now, else wait up to 1.5 s more.
    inf = statistics.median(res[1][2])
    tp = sum(p >= .5 for p, _ in done) / len(done)
    wait = tp * (0.2 + inf) + (1 - tp) * (0.2 + 1.5)
    cut_off = sum(p >= .5 for p, _ in notdone) / len(notdone)
    print(f"\nSmart Turn policy: mean wait after a finished sentence {wait:.2f}s (today 0.6 s + 0.4 s debounce = 1.0 s); "
          f"cuts you off at a mid-sentence pause {cut_off:.0%} of the time")


if __name__ == "__main__":
    main()
