"""Time to first audio: speaking a reply's first clause versus its whole first sentence,
per voice, through the plugin's own engines. The openers are real first sentences from
voice sessions.

    ~/.local/share/voice-mode/venv/bin/python bench/first_audio.py
"""
import re, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helper"))
from voice.processors.speakable import speakable
from voice.services.tts_service import TTSSwitcher

OPENERS = [
    "Sure, the basics are simple, and you can do most of it without thinking about it.",
    "Found it, your website is a Jekyll site hosted on GitHub Pages.",
    "Yes, I can hear you clearly, and voice mode is on.",
    "That sounds like my own voice came back through the mic.",
    "Agreed, this is a known step in speech pipelines called text normalization.",
    "I think you mean idempotent operations, and I'm going with that.",
    "Good, then it worked, and it heard you correctly both times.",
    "The numbers from this session are in, and they look good.",
    "Okay, so the slow part is your laptop, not voice mode.",
    "Right, found it, transcription took almost four seconds.",
]


def first_clause(sentence):
    """Up to the first comma, if that leaves a short opener."""
    head = re.split(r"(?<=,)\s", sentence, maxsplit=1)[0]
    return head if head != sentence and len(head.split()) <= 6 else sentence


def main():
    for voice in ["scarlet", "ryan"]:
        tts = TTSSwitcher(voice, 1.05)
        tts.synthesize("Warming up the model once.")
        whole, clause = [], []
        for s in OPENERS:
            s = speakable(s)
            t = time.perf_counter(); tts.synthesize(s); whole.append(time.perf_counter() - t)
            t = time.perf_counter(); tts.synthesize(first_clause(s)); clause.append(time.perf_counter() - t)
        print(f"{voice:8} first audio: whole sentence median {statistics.median(whole):.2f}s (max {max(whole):.2f}), "
              f"first clause median {statistics.median(clause):.2f}s (max {max(clause):.2f})")


if __name__ == "__main__":
    main()
