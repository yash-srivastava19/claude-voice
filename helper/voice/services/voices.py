"""The voice catalog, for every engine: a voice is a name, the engine that speaks it,
and a ref that only that engine understands.

Add a built-in voice with one line in PRESETS. Add your own by dropping it into
USER_VOICES; its type picks the engine:

    name.npy, name.pt    a Kokoro style vector        -> kokoro
    name/                a Piper model folder         -> piper  (sherpa-onnx's tarball layout)
    name.sh              any program: text on stdin, 16-bit WAV on stdout -> command

Anything else is read as a Kokoro voice id or mix ("af_sky", "af_heart:0.6+bf_emma:0.4").
"""

import os
from dataclasses import dataclass
from pathlib import Path

from voice.config import USER_VOICES


@dataclass(frozen=True)
class VoiceSpec:
    name: str
    engine: str
    ref: str
    about: str = ""


# Picked from Kokoro's best-rated English voices. A "+" joins voices into a mix,
# each weighted by its ":<share>".
PRESETS = [
    VoiceSpec("heart", "kokoro", "af_heart", "warm, US female"),
    VoiceSpec("bella", "kokoro", "af_bella", "bright and expressive, US female"),
    VoiceSpec("nicole", "kokoro", "af_nicole", "soft, close to the mic, US female"),
    VoiceSpec("scarlet", "kokoro", "af_sky", "light and airy, US female (Kokoro's sky)"),
    VoiceSpec("emma", "kokoro", "bf_emma", "calm, British female"),
    VoiceSpec("michael", "kokoro", "am_michael", "steady, US male"),
    VoiceSpec("fenrir", "kokoro", "am_fenrir", "deep, US male"),
    VoiceSpec("puck", "kokoro", "am_puck", "lively, US male"),
    VoiceSpec("george", "kokoro", "bm_george", "warm, British male"),
    VoiceSpec("fable", "kokoro", "bm_fable", "storyteller, British male"),
    VoiceSpec("honey", "kokoro", "af_heart:0.6+af_bella:0.4", "mix of heart and bella"),
    VoiceSpec("velvet", "kokoro", "af_nicole:0.5+af_heart:0.5", "mix of nicole and heart"),
    # Piper: plainer than Kokoro, but about six times faster, so replies never stutter.
    VoiceSpec("ryan", "piper", "ryan-medium", "clear, US male (Piper, fastest)"),
    VoiceSpec("lessac", "piper", "lessac-medium", "clear, US female (Piper, fastest)"),
]
BY_NAME = {v.name: v for v in PRESETS}


def _user_engine(path: Path):
    if path.suffix in (".npy", ".pt"):
        return "kokoro"
    if path.suffix == ".sh" and os.access(path, os.X_OK):
        return "command"
    if path.is_dir() and any(path.glob("*.onnx")) and (path / "tokens.txt").exists():
        return "piper"
    return None


def user_voices() -> dict[str, VoiceSpec]:
    if not USER_VOICES.is_dir():
        return {}
    found = {}
    for f in sorted(USER_VOICES.iterdir()):
        engine = _user_engine(f)
        if engine:
            found.setdefault(f.stem, VoiceSpec(f.stem, engine, str(f)))
    return found


def resolve(name: str) -> VoiceSpec:
    """The voice a name means: yours first, then a preset, then a raw Kokoro id or mix."""
    name = name.strip()
    mine = user_voices().get(name)
    if mine:
        return mine
    preset = BY_NAME.get(name.lower())
    if preset:
        return preset
    return VoiceSpec(name, "kokoro", name)


def list_voices():
    from voice.services.kokoro.voices import kokoro_ids
    return {
        "presets": [{"name": v.name, "voice": v.ref, "engine": v.engine, "about": v.about} for v in PRESETS],
        "kokoro": kokoro_ids(),
        "yours": [{"name": v.name, "file": v.ref, "engine": v.engine} for v in user_voices().values()],
    }
