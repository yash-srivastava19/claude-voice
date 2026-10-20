# claude-voice

Talk to Claude Code out loud. It answers in speech; code and tables stay on screen.
Runs fully local (whisper.cpp + Kokoro). No API keys.

## Install

Needs Claude Code 2.1.286+, macOS or Linux. On macOS, `brew install sox` first.

1. Turn on plugin hooks (early access) in `~/.claude/settings.json`:

   ```json
   "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1" }
   ```

2. In Claude Code:

   ```
   /plugin marketplace add yash-srivastava19/claude-voice
   /plugin install voice@claude-voice
   ```

3. Restart Claude Code and type `/talk start`. The first start downloads the
   models (~500 MB, a few minutes).

## Use

- Just talk. Use headphones.
- Talk over Claude to interrupt; say "stop" to cancel.
- After 2 minutes of quiet it sleeps; say "hey Claude" to wake it.
- `/talk stop` to end.
- Log: `~/.local/share/voice-mode/voiced.log`

## Voices

```
/talk voices                 list them
/talk voice heart            switch (you hear a short sample)
/talk voice af_heart:0.6+bf_emma:0.4    mix two voices
/talk speed 1.1              faster or slower (0.5 to 2.0)
```

Built in: heart, bella, nicole, emma, michael, fenrir, puck, george, fable, plus
the mixes honey and velvet. Any of Kokoro's English voices works by its id too.
Your choice is saved; it is also under `/config`.

### Add your own voice

Put a file in `~/.local/share/voice-mode/voices/`, then `/talk voice <file name>`:

- `<name>.npy`: a Kokoro voice. To make one that sounds like a recording, use
  [KVoiceWalk](https://github.com/RobViren/kvoicewalk) (needs a 20 to 30 second
  clip; takes hours, a GPU helps a lot). It writes a `.pt` file; convert it once:

  ```
  uv run --with torch --with numpy python -c "import torch, numpy; numpy.save('name.npy', torch.load('name.pt').numpy())"
  ```

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

To get updates automatically: `/plugin` → Marketplaces → claude-voice → Enable auto-update.
