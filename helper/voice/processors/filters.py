"""Pure functions over transcripts and sentences: what to keep, what to drop."""

import re
from difflib import SequenceMatcher

from voice.config import CHUNK_WORDS

# Things whisper "hears" in silence or noise.
HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "so", "uh", "um", "hmm", "bye bye", "thank you very much", "thanks a lot",
    "see you in the next video", "ill see you in the next video", "i will see you in the next video",
    "please subscribe", "like and subscribe", "love you", "i love you",
}

WAKE_PHRASE = re.compile(
    r"^\s*(?:(?:hey|hay|hi|okay|ok|yo|eh|a)[\s,-]+)?"
    r"(?:claude|claud|clod|clord|clode|clawed|clyde|cloud|clone)\b[\s,.!?]*",
    re.IGNORECASE,
)
# How the small model mishears "hey Claude" through a headset mic: "A-Clord", "a clone",
# "OK cloud". A greeting-like word plus a cl-word, or a cl-word alone, opens the segment.
GREETING_LIKE = {"hey", "hay", "hi", "okay", "ok", "yo", "eh", "a", "hei", "heh"}
CLAUDE_LIKE = re.compile(r"^[ck]l(?:a|o|au|aw|ou|or|y)[a-z]{0,3}$")
NOT_CLAUDE = {"close", "closed", "clean", "clear", "class", "clock", "cloth", "clue", "claim",
              "clap", "clip", "club", "claws", "clay", "clan", "clam", "clots"}

ECHO_MATCH = 0.6  # share of heard words that sit in runs of what was playing
# Short interruptions that always get through, even when Claude just said the same word.
STOP_PHRASES = {"stop", "wait", "hold on", "cancel", "never mind", "shut up"}
# Listening noises. Said over Claude they mean "go on", not "stop", so the reply resumes;
# said while Claude is quiet they are an answer and go through.
BACKCHANNEL_WORDS = {"uh", "huh", "uhhuh", "mhm", "mhmm", "mm", "hmm", "mmhmm", "aha", "ah", "oh",
                     "right", "okay", "ok", "yeah", "yep", "yup", "sure", "cool", "nice", "alright", "gotcha"}
BACKCHANNEL_PHRASES = {"got it", "i see", "go on", "makes sense", "fair enough", "all right"}


def heard_wake(text: str) -> bool:
    """Lenient wake test for the small model's transcript of a segment's opening."""
    tokens = re.findall(r"[a-z]+", text.lower())
    def claude_like(token):
        return bool(CLAUDE_LIKE.match(token)) and token not in NOT_CLAUDE
    if tokens and claude_like(tokens[0]):
        return True
    return len(tokens) > 1 and tokens[0] in GREETING_LIKE and claude_like(tokens[1])


def strip_wake(text: str):
    """Returns (is_wake, text after the wake phrase) for a transcript."""
    m = WAKE_PHRASE.match(text)
    if not m:
        return False, text
    return True, text[m.end():].strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def is_echo(transcript: str, playing: list[str]) -> bool:
    """True when a transcript is mostly the sentences that were playing as it was heard.

    The caller passes only sentences whose playback overlapped the utterance, so speech
    while nothing played is never echo. Echo comes back as runs of Claude's words in
    Claude's order; a reply reuses single words ("can you hear me?" after "I can hear
    you"), so only runs count. Stop phrases always get through.
    """
    heard = [stem(w) for w in words(transcript)]
    if not heard or not playing or " ".join(words(transcript)) in STOP_PHRASES:
        return False
    spoken = [stem(w) for sentence in playing for w in words(sentence)]
    blocks = SequenceMatcher(None, heard, spoken, autojunk=False).get_matching_blocks()
    if len(heard) < 3:
        return any(block.size == len(heard) for block in blocks)
    # Only runs of two or more words count: echo arrives as stretches of Claude's
    # sentence, a reply picks single words out of it.
    found = sum(block.size for block in blocks if block.size >= 2)
    return found / len(heard) >= ECHO_MATCH


def is_backchannel(transcript: str) -> bool:
    """True for a short listening noise such as "uh-huh", "right" or "okay, got it"."""
    heard = words(transcript.replace("-", ""))
    if not heard or len(heard) > 4:
        return False
    phrase = " ".join(heard)
    if phrase in BACKCHANNEL_PHRASES:
        return True
    rest = phrase
    for p in BACKCHANNEL_PHRASES:
        rest = rest.replace(p, " ")
    return all(w in BACKCHANNEL_WORDS for w in rest.split())


def chunk_sentence(text: str) -> list[str]:
    """Splits a long sentence at clause punctuation into chunks of at least 4 words."""
    if len(text.split()) <= CHUNK_WORDS:
        return [text]
    parts = re.split(r"(?<=[,;:])\s+", text)
    chunks, current = [], ""
    for part in parts:
        current = f"{current} {part}".strip()
        if len(current.split()) >= 4:
            chunks.append(current)
            current = ""
    if current:
        if chunks:
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)
    return chunks


def split_opener(text: str) -> list[str]:
    """A reply's first sentence, cut so the first sound comes sooner: its opening clause
    ("Sure," "Found it,") on its own when that is short, else clause-sized chunks.
    The opener costs Kokoro 0.3 s against 1.1 s for the whole sentence (bench/first_audio.py)."""
    head, sep, rest = text.partition(", ")
    if sep and len(head.split()) <= 6 and len(rest.split()) >= 3:
        return [head + ",", rest]
    return chunk_sentence(text)


def clean_transcript(text: str) -> str:
    """Strip whisper's non-speech tags and drop known hallucinations."""
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    bare = re.sub(r"[^a-z ]", "", text.lower()).strip()
    if not bare or bare in HALLUCINATIONS:
        return ""
    return text
