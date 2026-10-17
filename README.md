# claude-voice

Talk to Claude Code out loud. It answers in speech; code and tables stay on screen.
Runs fully local (whisper.cpp + Kokoro). No API keys.

## Install

Needs Claude Code 2.1.286+, macOS or Linux.

```sh
git clone https://github.com/yash-srivastava19/claude-voice
cd claude-voice
./install.sh
```

Downloads ~500 MB of models once. On macOS, `brew install sox` first.

## Run

```sh
CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir /path/to/claude-voice
```

Then type `/talk start`. Use headphones. `/talk stop` to end.

- Talk over Claude to interrupt; say "stop" to cancel.
- After 2 minutes of quiet it sleeps; say "hey Claude" to wake it.
- Log: `~/.local/share/voice-mode/voiced.log`

The plugin hooks API is early access and may change between Claude Code releases.
