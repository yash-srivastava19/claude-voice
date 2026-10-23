"""Holds the balanced power profile while voice is on (see POWER_FILE in config)."""

import subprocess

from voice.config import POWER_FILE, VOICE_POWER
from voice.log import log


def power_profile(set_to=None):
    cmd = ["powerprofilesctl", "set", set_to] if set_to else ["powerprofilesctl", "get"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return res.stdout.strip() if res.returncode == 0 else ""


def hold_power():
    if VOICE_POWER == "off":
        return
    now = power_profile()
    if now == "power-saver":
        if not POWER_FILE.exists():
            POWER_FILE.write_text(now)
        power_profile(VOICE_POWER)
        log(f"power: {now} -> {power_profile() or 'unchanged'} while voice is on")


def release_power():
    if not POWER_FILE.exists():
        return
    before = POWER_FILE.read_text().strip()
    if power_profile() == VOICE_POWER:  # left alone if you changed it yourself meanwhile
        power_profile(before)
        log(f"power: restored {before}")
    POWER_FILE.unlink(missing_ok=True)
