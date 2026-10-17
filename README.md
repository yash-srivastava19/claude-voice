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
