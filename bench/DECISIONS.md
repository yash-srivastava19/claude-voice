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

## Across machines

Same harness (`run.py all`), same 181 clips, CPU only. The laptop row is a quick run
(40 clips, 1 round); AWS is the full set, 2 rounds, load under 4.

| | Laptop: i7-1255U, 12 threads | AWS c6a.4xlarge: EPYC 7R13, 16 vCPU |
|---|---|---|
| Parakeet errors (accented / Indian / read / jargon) | 18.0 / 16.7 / 2.2 / 6.3 % (subset) | 17.2 / 10.5 / 2.1 / 7.3 % (identical to the laptop's full-set run) |
| Parakeet, a turn of 5 s or less | 0.43 s (p90 0.58) | 0.14 s (p90 0.21) |
| Scarlet first clause / first sentence | 0.30 / 1.19 s | 0.19 / 0.62 s |
| Scarlet RTF alone / with speech to text | 0.34 / 0.36 | 0.18 / 0.34 |
| Ryan first sentence, RTF | 0.15 s, 0.05 | 0.17 s, 0.06 |
| Speech to text while TTS runs, priority on | 0.61 s | 0.23 s |

Same defaults hold on both: Parakeet, Kokoro, opener split, TTS at lower priority. On
the 16-core machine nothing comes near stuttering; on the laptop the priority change is
what keeps transcription fast. AWS memory numbers from this run are not used
(ru_maxrss inherited the parent's peak; fixed since).

## Open

- GPU (CUDA) and Apple Silicon numbers: `run.py --provider cuda|coreml` on those machines.
- Fillers: on by default for the A/B; kept if `turn_report.py` shows less dead air and the person prefers them.
- Other harnesses. The daemon is harness-neutral (Unix socket); each needs a thin adapter
  that speaks the reply and submits the person's words into the open session.
  - Codex 0.150: both directions verified live in tmux. The `Stop` hook carries
    `last_assistant_message`; `codex queue --thread <session_id> --message <text>` lands in
    the running TUI and starts a turn. New hooks need one trust prompt.
  - OpenCode 1.18: the plugin API has streamed text parts (`message.part.updated`) and
    `client.tui.appendPrompt` + `submitPrompt`. Not verified: no model answered on this
    laptop (hosted models errored, the local one timed out).
  - Grok 1.0: the `Stop` hook carries `lastAssistantMessage`; no way found yet to submit
    into a running session from outside.
  - Before any adapter: move the markdown-to-speech filter (hooks/speech.ts) where every
    adapter can use it, with one shared set of test cases.
