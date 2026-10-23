# Voice decisions and the numbers behind them

Every default in the plugin traces to a row here. Measured on a 12th-gen i7-1255U laptop
(2 performance + 8 efficiency cores, 16 GB, no GPU), Linux, on battery unless noted,
with background load 1 to 9. Speech data: 91 EdAcc clips (real conversation, 11 accents,
13 Indian English), 30 LibriSpeech test-clean clips, 60 synthetic jargon sentences
(`data.py`). Word error rate is lower-is-better; RTF is synthesis time over audio time,
and above 1 the voice stutters.

## Kept

| Decision | Evidence | Script |
|---|---|---|
| Parakeet TDT 0.6B v2 int8 for speech to text | errors accented 28.0% -> 17.2%, Indian English 25.7% -> 10.5%, read 5.3% -> 2.1%, jargon 10.5% -> 7.3% against whisper base; typical turn 1.63 s -> 1.22 s; 1.2 GB RAM vs 0.33 GB | `run.py stt` |
| Parakeet on 6 threads | 0.95 s alone and 1.14 s with TTS running, against 1.31 s and 1.75 s on 8 | contention |
| Piper voices (ryan, lessac) offered | RTF 0.12 alone and 0.24 under load vs Kokoro 0.70 and 1.43; first sentence 0.44 s vs 2.54 s; clarity 2.2% garble vs 1.1% | `run.py tts` |
| Kokoro stays the default voice engine | it sounds more natural (a listening call); the two rows below remove its lag | listening |
| Speak a reply's opening clause first | in the real daemon, first sound 1.01 s (0.83 to 1.31) -> 0.29 s (0.25 to 0.61), 9 replies per build over 3 alternating rounds; a 0.31 s beat after the opener | `live.py` |
| Text to speech at lower priority | in the real daemon, your words while a reply synthesizes 0.43 s -> 0.30 s (0.25 to 0.38); with it off the new build is back at 0.45 s, so priority is the cause. The reply is paused then, so its slower synthesis is not heard | `live.py` |
| Text normalization before speech | 8 tricky sentences, heard back the way a person says them: 66% -> 82% of words | `tests/test_speakable.py` |
| Treat "aha", "ah", "oh", "mhmm" as listening noises | whisper heard "uh-huh, right" as "Aha, right." and cut the reply | `tests/test_pipeline.py` |

## Rejected

| Idea | Evidence | Script |
|---|---|---|
| whisper short audio window | 4x faster but read-speech errors 5.3% -> 17.7% | `run.py stt` (earlier run) |
| vocabulary hint for whisper | worse in every group, jargon included | `run.py stt` (earlier run) |
| Moonshine (v1 and 2026) | fast, but accented errors 30% to 62% | `run.py stt` (earlier run) |
| Streaming speech to text (transcribe at pauses) | final text 0.75 s -> 0.37 s, errors 12.4% -> 17.6%; the mildest version saves 0.14 s for +1.7 points | `experiments/streaming_stt.py` |
| Smart Turn v3.2 end-of-turn model | right on 38% of finished sentences and 59% of mid-sentence pauses; waits longer than today and cuts off 41% of pauses | `experiments/turn_detect.py` |
| Dropping the 0.4 s submit debounce | at least 14% of 1,135 real utterances were followed by more speech within 1 s | `experiments/debounce.py` |
| Kokoro int8 instead of fp32 | 3.4x slower on this CPU: RTF median 1.08 (1.02 to 1.15) vs 0.32 (0.30 to 0.41), 5 alternating rounds, 15 sentences each. An earlier run that put int8 16% ahead had the two sides set up differently | `run.py tts`, controlled rerun |
| Kokoro on OpenVINO | the model has a dynamic-rank op the OpenVINO CPU plugin rejects | `run.py tts` (earlier run) |
| Capping Kokoro's threads | 4 threads slower than default alone and under load | contention |

## Open

- GPU (CUDA) and Apple Silicon numbers: `run.py --provider cuda|coreml` on those machines.
- Fillers: on by default for the A/B; kept if `turn_report.py` shows less dead air and the person prefers them.
