"""The transcript filters: pure functions, no models or audio devices needed."""
import pytest

from conftest import voice

f = voice.filters


@pytest.mark.parametrize("text,expected", [
    ("Uh-huh.", True), ("Mm-hmm", True), ("Right.", True), ("Okay, got it.", True),
    ("Yeah, makes sense", True), ("I see", True), ("okay okay", True),
    # How whisper actually spells "uh-huh, right" through a mic.
    ("Aha, right.", True), ("Mhmm.", True), ("Oh, okay.", True), ("Oh wait, no.", False),
    ("Stop.", False), ("Wait, no.", False), ("No.", False),
    ("Right, but why is it slow?", False), ("Okay so change the tail", False), ("", False),
])
def test_backchannel(text, expected):
    assert f.is_backchannel(text) is expected


@pytest.mark.parametrize("heard,playing,expected", [
    # Claude's own sentence coming back through the mic.
    ("so the first check is easy", ["So the first check is easy: if I am talking, the mic is paused."], True),
    # A reply that reuses single words of what was playing.
    ("can you hear me", ["I can hear you fine now."], False),
    # Nothing was playing: never echo.
    ("so the first check is easy", [], False),
    # Stop phrases always get through, even if Claude just said them.
    ("stop", ["Say stop to interrupt."], False),
    # Short transcripts must match a whole run.
    ("mic paused", ["the mic is paused"], False),
    ("is paused", ["the mic is paused"], True),
])
def test_echo(heard, playing, expected):
    assert f.is_echo(heard, playing) is expected


@pytest.mark.parametrize("text,expected", [
    ("Hey Claude", True), ("hey clod", True), ("Claude, are you there?", True),
    ("Hey close the door", False), ("clean up the repo", False), ("hello there", False),
])
def test_heard_wake(text, expected):
    assert f.heard_wake(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("Hey Claude, run the tests.", (True, "run the tests.")),
    ("Run the tests.", (False, "Run the tests.")),
])
def test_strip_wake(text, expected):
    assert f.strip_wake(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Thank you.", ""), (" [BLANK_AUDIO] ", ""), ("(music)", ""), ("you", ""),
    ("Run the tests [laughs] again.", "Run the tests again."),
    ("Thank you for fixing it.", "Thank you for fixing it."),
])
def test_clean_transcript(text, expected):
    assert f.clean_transcript(text) == expected


def test_chunk_sentence_keeps_short_and_splits_long():
    assert f.chunk_sentence("Short one.") == ["Short one."]
    long = ("The first check is easy, if I am talking the mic is paused, then it waits half a second "
            "because the speakers keep ringing, and after that it just listens.")
    chunks = f.chunk_sentence(long)
    assert " ".join(chunks) == long and len(chunks) > 1 and all(len(c.split()) >= 4 for c in chunks)
