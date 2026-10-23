"""Text normalization: what the voice model is given to say."""
import pytest

from voice.processors.speakable import speakable


@pytest.mark.parametrize("text,said", [
    ("It took 0.8 s.", "It took 0 point eight seconds."),
    ("about 0.36s for 2.3s of audio", "about 0 point three six seconds for 2 point three seconds of audio"),
    ("It's 4x faster.", "It's 4 times faster."),
    ("music from the 1990s", "music from the 1990s"),
    ("Errors fell 26% to 10.5%.", "Errors fell 26 percent to 10 point five percent."),
    ("It uses 1182 MB.", "It uses 1182 megabytes."),
    ("Upgrade to v0.2.4 first.", "Upgrade to version 0 point 2 point 4 first."),
    ("It lives in yash_blog, on yash-sri.xyz.", "It lives in yash blog, on yash sri dot x y z."),
    ("Open register.tsx now.", "Open register dot t s x now."),
    ("See example.com and config.json.", "See example dot com and config dot jason."),
    ("from 2.5 s → 0.8 s", "from 2 point five seconds to 0 point eight seconds"),
    ("~3 seconds", "about 3 seconds"),
    # Plain words and integers are left alone; hyphens become spaces.
    ("Run the tests again.", "Run the tests again."),
    ("I merged 12 pull requests.", "I merged 12 pull requests."),
    ("a well-known pair-programming trick", "a well known pair programming trick"),
])
def test_speakable(text, said):
    assert speakable(text) == said
