"""The fixed speech test set: built the same way on every machine, so reports compare.

    python bench/data.py      (run.py calls it when the set is missing)

181 clips in $BENCH/set/ with manifest.json:
  91 accented   real conversational speech from EdAcc's test split, 8 per accent, 13 Indian English
  30 read       LibriSpeech test-clean
  60 jargon     JARGON read by four Kokoro voices (synthetic: fair for latency, kind on accuracy)
"""

import collections
import io
import json
import os
import random
import re
import urllib.request
from pathlib import Path

BENCH = Path(os.environ.get("BENCH", Path.home() / ".cache/claude-voice-bench"))
SET = BENCH / "set"
RATE = 16000
SOURCES = {
    "edacc0.parquet": "https://huggingface.co/api/datasets/edinburghcstr/edacc/parquet/default/test/0.parquet",
    "edacc1.parquet": "https://huggingface.co/api/datasets/edinburghcstr/edacc/parquet/default/test/1.parquet",
    "libri.parquet": "https://huggingface.co/api/datasets/openslr/librispeech_asr/parquet/all/test.clean/0.parquet",
}
JARGON = [
    "Run the tests again.",
    "Open the Claude voice plugin.",
    "Does this work with any harness, not just Claude Code?",
    "Stop.",
    "Check the PipeWire echo cancel config.",
    "Why is Kokoro slower than whisper on this laptop?",
    "Rebase the branch onto main and push it.",
    "Can you grep the voice log for mic stream ended?",
    "Add a pytest for the backchannel check.",
    "Okay, got it. Now what about the Mac?",
    "Make funes run at idle priority so it doesn't slow down voice mode.",
    "Show me the diff for register dot tsx before you commit.",
    "Hey Claude, are you there?",
    "The int8 model sounds a bit worse to me, can we compare them side by side?",
    ("So what I want is for voice mode to feel as fast as typing on this laptop, "
     "and then on a faster machine like the Mac it should feel instant."),
]
JARGON_VOICES = ["am_michael", "af_heart", "hf_alpha", "hm_omega"]  # the last two Hindi-accented


def to16k(x, sr):
    import numpy as np
    if x.ndim > 1:
        x = x.mean(1)
    if sr != RATE:
        idx = np.linspace(0, len(x) - 1, int(len(x) * RATE / sr))
        x = np.interp(idx, np.arange(len(x)), x)
    return x.astype(np.float32)


def fetch():
    (BENCH / "data").mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = BENCH / "data" / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, path)


def clean(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>|\[[^\]]*\]|\([^)]*\)", " ", text)).strip()


def build():
    import pyarrow.parquet as pq
    import soundfile as sf
    from kokoro_onnx import Kokoro

    fetch()
    SET.mkdir(parents=True, exist_ok=True)
    rng, items = random.Random(7), []

    def add(name, audio, text, group):
        sf.write(SET / f"{name}.wav", audio, RATE, subtype="PCM_16")
        items.append({"file": f"{name}.wav", "text": text, "group": group, "dur": round(len(audio) / RATE, 2)})

    by_accent = collections.defaultdict(list)
    for f in ("edacc0.parquet", "edacc1.parquet"):
        for row in pq.read_table(BENCH / "data" / f).to_pylist():
            text = clean(row["text"])
            if len(text.split()) >= 4 and row["accent"] != "Don't know":
                by_accent[row["accent"]].append((row, text))
    n = 0
    for accent, rows in by_accent.items():
        rng.shuffle(rows)
        want = 13 if accent == "Indian English" else 8
        for row, text in rows:
            x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            if 1.5 <= len(x) / sr <= 15:
                add(f"edacc_{n:03}", to16k(x, sr), text, "accented:" + accent)
                n, want = n + 1, want - 1
            if not want:
                break

    rows = pq.read_table(BENCH / "data" / "libri.parquet").to_pylist()
    rng.shuffle(rows)
    k = 0
    for row in rows:
        x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if 2 <= len(x) / sr <= 15:
            add(f"libri_{k:03}", to16k(x, sr), row["text"], "read:librispeech")
            k += 1
        if k == 30:
            break

    models = Path.home() / ".local/share/voice-mode/models"
    tts = Kokoro(str(models / "kokoro-v1.0.onnx"), str(models / "voices-v1.0.bin"))
    j = 0
    for voice in JARGON_VOICES:
        for sentence in JARGON:
            pcm, sr = tts.create(sentence, voice=voice, speed=1.0, lang="en-us")
            add(f"jargon_{j:03}", to16k(pcm, sr), sentence, f"jargon:{voice}")
            j += 1

    (SET / "manifest.json").write_text(json.dumps(items, indent=0))
    print(f"{len(items)} clips in {SET}")


def load():
    """The manifest, building the set first if this machine has none."""
    if not (SET / "manifest.json").exists():
        build()
    return json.loads((SET / "manifest.json").read_text())


if __name__ == "__main__":
    build()
