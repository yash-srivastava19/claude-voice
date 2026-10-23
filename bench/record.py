"""Record yourself reading known sentences, for measuring transcription accuracy.

    /talk stop          (in Claude Code first, or every sentence becomes a prompt)
    ~/.local/share/voice-mode/venv/bin/python bench/record.py

Press Enter to start each sentence and Enter again when you're done. Records from
the same mic the daemon uses. Clips go to bench/real/ and are never committed.
"""
import json, os, shutil, subprocess, sys
from pathlib import Path

OUT = Path(__file__).with_name("real")
SENTENCES = [
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


def recorder(path):
    if sys.platform == "darwin":
        return ["rec", "-q", "-r", "16000", "-c", "1", "-b", "16", str(path)]
    src = os.environ.get("VOICE_MIC_SOURCE", "claude_voice_mic")
    if subprocess.run(["pw-cli", "info", src], capture_output=True).returncode != 0:
        src = None  # no echo-cancel source: use the default mic
    return ["pw-record", *(["--target", src] if src else []), "--rate", "16000", "--channels", "1",
            "--format", "s16", str(path)]


def main():
    OUT.mkdir(exist_ok=True)
    prompts = {}
    print(f"{len(SENTENCES)} sentences. Read each one naturally, at your normal pace.\n")
    for i, text in enumerate(SENTENCES, 1):
        path = OUT / f"{i:02}.wav"
        print(f"[{i}/{len(SENTENCES)}]  {text}")
        input("  Enter to start… ")
        rec = subprocess.Popen(recorder(path), stderr=subprocess.DEVNULL)
        input("  recording, Enter when done… ")
        rec.terminate()
        rec.wait()
        prompts[path.name] = text
        (OUT / "prompts.json").write_text(json.dumps(prompts, indent=1))
        print()
    print(f"Saved {len(prompts)} clips in {OUT}")


if __name__ == "__main__":
    main()
