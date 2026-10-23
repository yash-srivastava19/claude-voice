"""Text normalization: rewrites a sentence the way a person would say it, before text
to speech sees it. Voice models read "0.8" as "zero eight", spell out "yash_blog"
and stumble on "yash-sri.xyz"; these rules turn them into "zero point eight", "yash
blog" and "yash sri dot xyz". Pure and fast: one regex pass per rule, no model.

Each rule is (pattern, replacement), applied in order; add one with a test case in
tests/test_speakable.py.
"""

import re

DIGITS = "zero one two three four five six seven eight nine".split()
UNITS = {
    "ms": "milliseconds", "s": "seconds", "sec": "seconds", "min": "minutes", "h": "hours",
    "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes", "tb": "terabytes",
    "khz": "kilohertz", "hz": "hertz", "x": "times", "×": "times", "%": "percent",
}
SYMBOLS = {"→": " to ", "←": " from ", "≈": " about ", "~": " about ", "≤": " at most ", "≥": " at least ",
           "±": " plus or minus ", "&": " and ", "+": " plus "}
TLDS = "com|org|net|io|dev|ai|xyz|app|sh|co|me|so|md|py|ts|tsx|js|json|yml|yaml|toml|txt|log|wav|onnx"
# Endings people say as a word; every other one is spelled out: .xyz -> "dot x y z".
SAID_AS_WORDS = {"com": "com", "org": "org", "net": "net", "dev": "dev", "app": "app", "me": "me",
                 "json": "jason", "yaml": "yammel", "yml": "yammel", "toml": "tommel", "log": "log",
                 "wav": "wave", "onnx": "onyx"}


def _decimal(m):
    whole, frac = m.group(1), m.group(2)
    return f"{whole} point {' '.join(DIGITS[int(d)] for d in frac)}"


def _unit(m):
    if m.group(2) == "s" and re.fullmatch(r"(1[89]|20)\d0", m.group(1)):
        return m.group(0)  # the 1990s is a decade, not 1990 seconds
    return f"{m.group(1)} {UNITS[m.group(2).lower()]}"


def _version(m):
    return "version " + " point ".join(m.group(1).split("."))


def _dotted(m):
    *names, ending = m.group(0).split(".")
    said = SAID_AS_WORDS.get(ending.lower()) or " ".join(ending.lower())
    return " dot ".join([*names, said])


RULES = [
    # v0.2.3 -> version 0 point 2 point 3 (before decimals eat "0.2")
    (re.compile(r"\bv(\d+(?:\.\d+){1,3})\b"), _version),
    # yash-sri.xyz, register.tsx, data.json -> yash-sri dot x y z, register dot t s x, data dot jason
    (re.compile(rf"\b[\w-]+(?:\.[\w-]+)*\.(?:{TLDS})\b", re.IGNORECASE), _dotted),
    # 1.5s, 300 ms, 40%, 4x -> 1.5 seconds, 300 milliseconds, 40 percent, 4 times
    (re.compile(r"(\d+(?:\.\d+)?)\s?(ms|sec|min|kb|mb|gb|tb|khz|hz|s|h|x|×|%)(?![\w])", re.IGNORECASE), _unit),
    # 0.8 -> 0 point eight; 12.05 -> 12 point zero five
    (re.compile(r"\b(\d+)\.(\d+)\b"), _decimal),
    # snake_case and kebab-case words -> spaces
    (re.compile(r"(?<=[A-Za-z0-9])[_](?=[A-Za-z0-9])"), " "),
    (re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])"), " "),
    # paths: ~/code/claude-voice -> home code slash claude voice
    (re.compile(r"~/"), "home "),
    (re.compile(r"(?<=\w)/(?=\w)"), " slash "),
]


def speakable(text: str) -> str:
    for pattern, repl in RULES:
        text = pattern.sub(repl, text)
    for sym, words in SYMBOLS.items():
        text = text.replace(sym, words)
    return re.sub(r"\s{2,}", " ", text).strip()
