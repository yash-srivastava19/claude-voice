# claude-voice

Talk to Claude Code out loud. It answers in speech; code and tables stay on screen.
Runs fully local on your CPU (Parakeet speech recognition, Kokoro voices). No API keys,
no GPU needed.

## Install

### What you need

| | macOS | Linux |
|---|---|---|
| Claude Code | 2.1.286+ | 2.1.286+ |
| Audio tools | `brew install sox` | `alsa-utils` (`arecord`, `aplay`) |
| curl | built in | your package manager |
| uv | installed for you if missing | installed for you if missing |
| Python | uses 3.11, 3.12 or 3.13 if you have one; otherwise uv downloads 3.13 | same |
| Disk | about 2 GB (1 GB of models, the rest a Python environment) | same |

Nothing else to set up by hand. On a slow connection, installing Python yourself
first (`brew install python@3.13`, or your distro's package) saves a 70 MB download.

### Steps

1. Turn on plugin hooks (early access) in `~/.claude/settings.json`:

   ```json
   "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1" }
   ```

2. In Claude Code:

   ```
   /plugin marketplace add yash-srivastava19/claude-voice
   /plugin install voice@claude-voice
   ```

3. Restart Claude Code and type `/talk start`. The first start sets everything up
   (about 1 GB of models, a few minutes). An interrupted download resumes on the
   next `/talk start`.

## Use

- Just talk.
- To cut a reply short: say "stop" (on speakers, only once the mic is listening; see
  below), press **Hush** in the voice band, or run `/talk hush`.
- While Claude works it says what it is doing ("Searching the web."), so a long
  pause never sounds like a dropped call. `/talk fillers off` turns that off.
- After 2 minutes of quiet it sleeps; say "hey Claude" to wake it.
- Voice follows one Claude Code session at a time. `/talk start` in another
  session moves it there, and the first session goes quiet.
- `/talk stop` to end.

### Speakers, headphones, and talking over Claude

On speakers the mic hears Claude too. How voice mode deals with that depends on
what your machine can do; `/talk start` says which one you got.

| Mode | When | Talk over Claude? |
|---|---|---|
| `aec` | Linux with the echo-cancel config below | yes: Claude's voice is subtracted from the mic |
| `gate` | everything else, including macOS speakers | no: the mic pauses while Claude speaks; press Hush, run `/talk hush`, or send a message |
| `open` | you set it, for headphones | yes: nothing is held back |

With headphones, put `"VOICE_ECHO": "open"` in the `env` block of
`~/.claude/settings.json` to talk over Claude. Don't use `open` on speakers: Claude
will hear itself and answer its own voice.

**Linux echo cancellation.** Copy the PipeWire config from this repo, then restart
PipeWire:

```
mkdir -p ~/.config/pipewire/pipewire.conf.d
cp pipewire/60-claude-voice-echo-cancel.conf ~/.config/pipewire/pipewire.conf.d/
systemctl --user restart pipewire pipewire-pulse
```

The next `/talk start` uses it. Delete the file and restart PipeWire to undo.

## Troubleshooting

- **Setup failed.** The message shows the step and the error. The full output is in
  `~/.local/share/voice-mode/install.log`. To retry, run the `install.sh` command the
  error message prints (it names the installed version's folder); finished steps are
  skipped and downloads resume.
- **It answers itself, or drops what you said.** Check the mode on the `audio:` line
  of the log. `open` on speakers causes this; unset `VOICE_ECHO`.
- **Daemon log:** `~/.local/share/voice-mode/voiced.log`, one line per thing heard,
  dropped or spoken.

## Voices

```
/talk voices                 list them
/talk voice heart            switch (you hear a short sample)
/talk voice af_heart:0.6+bf_emma:0.4    mix two voices
/talk speed 1.1              faster or slower (0.5 to 2.0)
```

Built in: heart, bella, nicole, scarlet, emma, michael, fenrir, puck, george, fable, plus
the mixes honey and velvet. Any of Kokoro's English voices works by its id too.

ryan and lessac are Piper voices: plainer, but about five times faster, for an older
or busy machine. Each downloads (about 80 MB) the first time you pick it.

Your choice is saved; it is also under `/config`.

### Add your own voice

Put a file in `~/.local/share/voice-mode/voices/`, then `/talk voice <file name>`:

- `<name>.npy`: a Kokoro voice. To make one that sounds like a recording, use
  [KVoiceWalk](https://github.com/RobViren/kvoicewalk) (needs a 20 to 30 second
  clip; takes hours, a GPU helps a lot). It writes a `.pt` file; convert it once:

  ```
  uv run --with torch --with numpy python -c "import torch, numpy; numpy.save('name.npy', torch.load('name.pt').numpy())"
  ```

- `<name>/`: a Piper voice, as a folder holding its `.onnx` model, `tokens.txt`
  and `espeak-ng-data` (the layout of
  [sherpa-onnx's Piper voices](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)).

- `<name>.sh`: any other speech engine. Make it executable; it gets the text on
  stdin (and `$VOICE_SPEED`) and must print a 16-bit WAV to stdout. For example,
  [Piper](https://github.com/OHF-Voice/piper1-gpl):

  ```sh
  #!/bin/sh
  f="${TMPDIR:-/tmp}/voice-$$.wav"
  piper -m ~/voices/en_US-amy-medium.onnx -f "$f" && cat "$f"; rm -f "$f"
  ```

Only use a voice you have the right to use (your own, or with the person's OK).

## Update

In Claude Code: `/plugin` → Marketplaces → claude-voice → Update marketplace.
Or from your shell: `claude plugin update voice@claude-voice`.
Then `/reload-plugins` (or restart). The next `/talk start` swaps in the new version.
It starts right away on what you already have; anything new (like a better speech
model) downloads in the background and is used from the start after that.

To get updates automatically: `/plugin` → Marketplaces → claude-voice → Enable auto-update.
