"""A preview of the redesigned voice band, played in the terminal.

    python3 bench/experiments/band_demo.py

One line per state, no second spinner (Claude Code draws its own), the person's words
cut to fit instead of wrapping.
"""
import math
import re
import shutil
import sys
import time

DIM, RESET = "\033[2m", "\033[0m"
GREEN, AMBER, CYAN, GREY = "\033[32m", "\033[33m", "\033[36m", "\033[90m"
BARS = "▁▂▃▄▅▆▇"


def fit(text, room):
    return text if len(text) <= room else text[: max(0, room - 1)] + "…"


def meter(t, loud, width=8):
    return "".join(BARS[min(6, int(loud * (0.5 + 0.5 * math.sin(t * 9 + i * 1.7)) * 7))] for i in range(width))


def pulse(t):
    return "◆" if int(t * 2) % 2 == 0 else "◇"


def line(state, t, said, quote):
    cols = shutil.get_terminal_size().columns
    if state == "listening":
        head, tail = f"{GREEN}●{RESET} listening", 'say something · "stop" interrupts'
    elif state == "hearing":
        head, tail = f"{GREEN}●{RESET} hearing  {GREEN}{meter(t, 0.9)}{RESET}", ""
    elif state == "thinking":
        head, tail = f"{AMBER}{pulse(t)}{RESET} thinking", f'"{quote}" · speak to add to this turn'
    else:
        head, tail = f"{CYAN}▶{RESET} speaking {CYAN}{meter(t, 0.6, 5)}{RESET}", said + ' · "stop" interrupts'
    visible = len(re.sub(r"\033\[[0-9;?]*[a-zA-Z]", "", head))
    return f"  {head}  {GREY}{fit(tail, cols - visible - 6)}{RESET}"


SCRIPT = [
    ("listening", 2.5, "", ""),
    ("hearing", 3.0, "", ""),
    ("thinking", 3.0, "", "Why is my blog not doing numbers? I posted twice this month and nothing moved"),
    ("speaking", 2.0, "Okay, let me look.", ""),
    ("thinking", 2.0, "", "Why is my blog not doing numbers? I posted twice this month and nothing moved"),
    ("speaking", 2.5, "Searching the web.", ""),
    ("speaking", 3.5, "I can't see your traffic numbers, but the repo shows a few reasons.", ""),
    ("hearing", 2.0, "", ""),
    ("listening", 2.0, "", ""),
]


def main():
    sys.stdout.write("\033[?25l")
    try:
        for _ in range(2):
            for state, secs, said, quote in SCRIPT:
                end = time.time() + secs
                while time.time() < end:
                    sys.stdout.write("\r\033[2K" + line(state, time.time(), said, quote))
                    sys.stdout.flush()
                    time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\r\033[2K\033[?25h\n")


if __name__ == "__main__":
    main()
