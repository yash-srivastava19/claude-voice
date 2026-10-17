import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { VoicePhase } from '../types'
import { SpeechSplitter } from './speech'

const isActive = atom({ plugin: 'voice', key: 'isActive' } as const, false)
const phase = atom({ plugin: 'voice', key: 'phase' } as const, 'off' as VoicePhase)
const heard = atom({ plugin: 'voice', key: 'heard' } as const, '')
const tick = atom({ plugin: 'voice', key: 'tick' } as const, 0)

const COMMAND = 'talk'
// The band above the prompt: one glyph, color and hint per phase, so listening,
// hearing, thinking and speaking read apart at a glance.
const PHASE_LOOK: Record<VoicePhase, { glyph: string; color?: string; hint: string }> = {
  off: { glyph: '○', hint: '' },
  starting: { glyph: '…', color: 'gray', hint: 'loading the voice models (first run downloads them)' },
  listening: { glyph: '●', color: 'green', hint: 'say something · "stop" interrupts' },
  hearing: { glyph: '◉', color: 'greenBright', hint: 'hearing you…' },
  thinking: { glyph: '◐', color: 'yellow', hint: 'working on it · speak to add to this turn' },
  speaking: { glyph: '▶', color: 'cyan', hint: 'talk over me to interrupt' },
  asleep: { glyph: '☾', color: 'magenta', hint: 'say "hey Claude" or type' },
}
const SPINNER = ['◐', '◓', '◑', '◒']
const POLL_MS = 250
const KEEP_AWAKE_MS = 20_000
// Once no speech segment is open, wait this long before submitting. Segments
// still being transcribed hold the prompt open on their own (see openSegments).
const SUBMIT_DEBOUNCE_MS = 400
const STOP_WORDS = /^(stop|wait|hold on|cancel|never mind|shut up)[.!]?$/i

const VOICE_CONTEXT = `Voice mode is on. The person is talking to you out loud through speech recognition, and every prose sentence you write is read aloud to them by a text-to-speech voice. Fenced code blocks, tables, headings and diagrams are shown on screen and are not read.

- Talk the way a senior engineer explains at a whiteboard: short, natural spoken sentences, first point first, conversational.
- Prose must sound right when heard. Keep file paths, URLs, inline code, identifiers, symbols and markdown emphasis out of prose: say "the auth middleware" and put the exact path or name in a code block or table.
- When something is better seen than heard (a diagram, a table of core concepts, a comparison, code, a command, exact names), put it on screen as a fenced code block or a markdown table, and point to it in speech, as in "Have a look at the diagram." Never read a table or code aloud.
- Draw diagrams only inside fenced code blocks; anything outside a fence is spoken.
- Lists are spoken too: keep each item one short spoken phrase, or move the list into a table.
- Keep spoken text short, a few sentences per reply, unless the person asks you to go deeper.
- Before tool calls, say at most one short line about what you are doing.
- Transcripts may mishear technical words; infer the intended meaning, and ask when you are unsure.`

// Per-load state: a reload evaluates the module afresh, and `session.start`
// resumes from $.state.
const v = {
  socket: '',
  poller: undefined as { cancel: () => void } | undefined,
  isPolling: false,
  failures: 0,
  gen: 0,
  sayChain: Promise.resolve() as Promise<unknown>,
  pending: [] as string[],
  // Speech segments heard but not yet transcribed: each `hearing` ends in one
  // `utterance` or `dropped`. The prompt waits while any is open.
  openSegments: 0,
  submitTimer: undefined as { cancel: () => void } | undefined,
  runningTurn: undefined as string | undefined,
  // Set by a barge-in or mid-turn speech: the rest of the step being streamed
  // is stale, so it stays on screen only. The next step speaks again.
  muteUntilNextStep: false,
  // Speech delivered into the running turn, and whether a step has started since
  // (a step that starts after the delivery is the one that reads it).
  delivered: [] as string[],
  isDeliveryRead: false,
  // Mid-turn speech the engine refused to append; it goes in when the turn ends.
  held: [] as string[],
  hasSpokenThisTurn: false,
  // The daemon sleeps after a quiet spell; a running turn keeps it awake.
  lastKeepAwake: 0,
}

async function call($: EngineInterface, method: 'GET' | 'POST', path: string, body?: object) {
  const res = await $.http.fetch(`http://voice${path}`, {
    method,
    socketPath: v.socket,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`voice daemon ${path}: HTTP ${res.status}`)
  return JSON.parse(res.text || '{}')
}

// First /talk start: install the Python environment and models (install.sh is
// idempotent and leaves a marker when done).
async function ensureInstalled($: EngineInterface) {
  const check = await $.process.run(['sh', '-c', '[ -f "${VOICE_HOME:-$HOME/.local/share/voice-mode}/.installed" ]'])
  if (check.exitCode === 0) return
  $.ui.toast('Voice: first run, downloading models (about 500 MB). This takes a few minutes.', { timeoutMs: 20_000 })
  const { exitCode, stderr } = await $.process.run(['sh', `${$.plugin.root}/install.sh`], { timeoutMs: 600_000 })
  if (exitCode !== 0) throw new Error(stderr.trim().split('\n').pop() || 'install failed; run install.sh in the plugin folder')
}

async function startDaemon($: EngineInterface) {
  const script = `${$.plugin.root}/helper/voiced.py`
  const { exitCode, stdout, stderr } = await $.process.run(
    ['sh', '-c', '"${VOICE_PYTHON:-$HOME/.local/share/voice-mode/venv/bin/python}" "$1" --start', 'sh', script],
    { timeoutMs: 30_000 },
  )
  if (exitCode !== 0) throw new Error(stderr.trim() || `voice daemon exited ${exitCode}`)
  v.socket = JSON.parse(stdout.trim().split('\n').pop() ?? '{}').socket
}

// One timeline: the mod's lines go into the daemon's log, on the daemon's clock.
function modLog($: EngineInterface, line: string) {
  void call($, 'POST', '/log', { line }).catch(() => {})
}

// The daemon bumps its generation on a barge-in, and a reload resets ours to 0;
// sentences tagged older than the daemon's generation are dropped, so keep up.
function syncGen(daemonGen: unknown) {
  if (typeof daemonGen === 'number') v.gen = Math.max(v.gen, daemonGen)
}

function say($: EngineInterface, text: string) {
  if (!v.hasSpokenThisTurn) {
    v.hasSpokenThisTurn = true
    modLog($, `first sentence of turn sent (gen ${v.gen})`)
  }
  const tagged = v.gen
  v.sayChain = v.sayChain.then(() => call($, 'POST', '/say', { text, gen: tagged })).catch(() => {})
}

// Stops speech now. Not chained behind queued sentences: the generation tag
// makes the daemon drop any of them that arrive after the flush.
function hush($: EngineInterface, why: string) {
  v.gen += 1
  void call($, 'POST', '/flush', { gen: v.gen, why }).catch(() => {})
}

async function submitPending($: EngineInterface) {
  v.submitTimer?.cancel()
  v.submitTimer = undefined
  if (v.pending.length === 0) return
  const text = v.pending.join(' ')
  v.pending = []
  if (v.runningTurn) {
    await deliverMidTurn($, text)
    return
  }
  modLog($, `submit: ${text}`)
  await update($, phase, () => 'thinking')
  await $.prompt.submit({ text, asUser: true })
}

// Speech while Claude works reaches the running turn at its next step, the way
// a message typed mid-turn does, instead of waiting for the whole turn to end.
async function deliverMidTurn($: EngineInterface, text: string) {
  hush($, 'mid-turn speech')
  v.muteUntilNextStep = true
  try {
    await appendSpeech($, text)
  } catch (err) {
    v.held.push(text)
    modLog($, `mid-turn append refused (${(err as Error).message}); holding for turn end: ${text}`)
    return
  }
  v.delivered.push(text)
  v.isDeliveryRead = false
  modLog($, `delivered mid-turn: ${text}`)
}

async function appendSpeech($: EngineInterface, text: string) {
  await $.session.append({
    message: {
      type: 'user',
      content: [{
        type: 'text',
        text: `The person just said this out loud while you were working: "${text}"\nRespond to it at your next step; if it changes the task, adjust.`,
      }],
    },
  })
  await $.session.append({ message: { type: 'system', content: [{ type: 'text', text: `voice, mid-turn: ${text}` }] } })
}

async function onUtterance($: EngineInterface, text: string) {
  await update($, heard, () => text)
  if (STOP_WORDS.test(text.trim())) {
    hush($, 'stop word')
    if (v.runningTurn) {
      modLog($, `stop word: aborting turn ${v.runningTurn}`)
      v.muteUntilNextStep = true
      await $.turn.abort({ turnId: v.runningTurn }).catch(() => {})
    }
    await update($, phase, () => 'listening')
    return
  }
  v.pending.push(text)
  scheduleSubmit($)
}

function scheduleSubmit($: EngineInterface) {
  v.submitTimer?.cancel()
  v.submitTimer = undefined
  // An open segment sends it when it closes; a running turn gets it mid-turn.
  if (v.openSegments > 0 || v.pending.length === 0) return
  v.submitTimer = $.clock.after(SUBMIT_DEBOUNCE_MS, () => void submitPending($))
}

async function poll($: EngineInterface) {
  if (v.isPolling) return
  v.isPolling = true
  try {
    if ((await read($, phase)) === 'thinking') await update($, tick, n => n + 1)
    const { events, gen } = await call($, 'GET', '/events')
    v.failures = 0
    syncGen(gen)
    if (v.runningTurn) {
      // A long tool run is not a quiet spell: keep the daemon awake while Claude works.
      const now = await $.clock.now()
      if (now - v.lastKeepAwake > KEEP_AWAKE_MS) {
        v.lastKeepAwake = now
        void call($, 'POST', '/wake', { why: 'turn running' }).catch(() => {})
      }
    }
    for (const ev of events as { kind: string; text?: string; gen?: number }[]) {
      if (ev.kind === 'hearing') {
        v.openSegments += 1
        v.submitTimer?.cancel() // still talking: hold the prompt
        await update($, phase, () => 'hearing')
      } else if (ev.kind === 'bargein') {
        syncGen(ev.gen)
        if (v.runningTurn) v.muteUntilNextStep = true
      } else if (ev.kind === 'utterance' && ev.text) {
        v.openSegments = Math.max(0, v.openSegments - 1)
        await onUtterance($, ev.text)
      } else if (ev.kind === 'dropped') {
        v.openSegments = Math.max(0, v.openSegments - 1)
        scheduleSubmit($)
        await update($, phase, () => (v.runningTurn ? 'thinking' : 'listening'))
      } else if (ev.kind === 'sleep') {
        modLog($, 'daemon asleep; say "hey Claude" or type to wake it')
        await update($, phase, () => 'asleep')
      } else if (ev.kind === 'wake') {
        await update($, phase, () => (v.runningTurn ? 'thinking' : 'listening'))
      } else if (ev.kind === 'speaking') {
        await update($, phase, () => 'speaking')
      } else if (ev.kind === 'idle') {
        await update($, phase, () => (v.runningTurn ? 'thinking' : 'listening'))
      }
    }
  } catch {
    v.failures += 1
    if (v.failures === 8) {
      // ~2 s without the daemon: it died or was shut down; bring it back once.
      await startDaemon($).catch(() => {})
    } else if (v.failures > 40) {
      await stop($)
      $.ui.toast('Voice mode stopped: the voice daemon is not responding. See ~/.local/share/voice-mode/voiced.log')
    }
  } finally {
    v.isPolling = false
  }
}

async function start($: EngineInterface) {
  await update($, phase, () => 'starting')
  await ensureInstalled($)
  await startDaemon($)
  await update($, isActive, () => true)
  await update($, phase, () => (v.runningTurn ? 'thinking' : 'listening'))
  v.failures = 0
  v.openSegments = 0
  v.poller?.cancel()
  v.poller = $.clock.every(POLL_MS, () => void poll($))
}

async function stop($: EngineInterface) {
  v.poller?.cancel()
  v.poller = undefined
  v.submitTimer?.cancel()
  v.pending = []
  await update($, isActive, () => false)
  await update($, phase, () => 'off')
  await update($, heard, () => '')
  if (v.socket) await call($, 'POST', '/shutdown').catch(() => {})
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    // Resume first: a reload must never leave voice on with nothing polling.
    if (await read($, isActive)) {
      // A reload or resume while voice was on: pick up where it was.
      await start($).catch(async () => {
        await stop($)
      })
      modLog($, 'mod loaded while voice was on (reload or resume)')
    }
    // `/voice` is a built-in command, so this one is `/talk`.
    await $.command
      .register({
        name: COMMAND,
        description: 'Talk to Claude hands-free: /talk start, /talk stop',
        argumentHint: '[start|stop]',
        immediate: true,
      })
      .catch(err => $.ui.toast(`voice: could not register /${COMMAND}: ${(err as Error).message}`))
    return next(e)
  })

  on('session.end', async ($, e, next) => {
    if (await read($, isActive)) await stop($)
    return next(e)
  })

  on('command.run', { command: COMMAND }, async ($, e) => {
    const arg = e.args.trim().toLowerCase()
    const isOn = await read($, isActive)
    const want = arg === 'start' || arg === 'on' ? true : arg === 'stop' || arg === 'off' ? false : !isOn

    if (want === isOn) return { text: isOn ? 'Voice mode is already on.' : 'Voice mode is already off.' }
    if (!want) {
      await stop($)
      return { text: 'Voice mode off.' }
    }
    try {
      await start($)
    } catch (err) {
      await stop($)
      return { text: `Voice mode could not start: ${(err as Error).message}` }
    }
    return { text: 'Voice mode on. Talk whenever you are ready; say "stop" to interrupt. /talk stop to end.' }
  })

  on('prompt.submit', async ($, e, next) => {
    if (!(await read($, isActive))) return next(e)
    hush($, 'new prompt') // a new prompt, typed or spoken, ends any reply still being read
    if (e.origin.kind === 'composer') void call($, 'POST', '/wake', { why: 'typed prompt' }).catch(() => {})
    return next({ ...e, context: [...(e.context ?? []), VOICE_CONTEXT] })
  })

  on('turn.start', async ($, e, next) => {
    if (!v.runningTurn) v.runningTurn = e.turnId // turn.start carries no agentId; keep the main turn
    v.hasSpokenThisTurn = false
    if (await read($, isActive)) {
      modLog($, `turn start ${e.turnId}`)
      await update($, phase, () => 'thinking')
    }
    return next(e)
  })

  on('turn.step', async function* ($, e, next) {
    if (e.agentId || !(await read($, isActive))) return yield* next(e)

    v.muteUntilNextStep = false
    if (v.delivered.length) v.isDeliveryRead = true
    const splitter = new SpeechSplitter()
    const speak = (sentences: string[]) => {
      if (v.muteUntilNextStep) return
      for (const s of sentences) say($, s)
    }
    const stream = next(e)
    for await (const chunk of stream) {
      yield chunk
      if (chunk.kind === 'text') speak(splitter.push(chunk.text, chunk.index))
    }
    speak(splitter.flush())
  })

  on('turn.complete', async ($, e, next) => {
    const result = await next(e)
    if (e.agentId) return result
    v.runningTurn = undefined
    v.muteUntilNextStep = false
    // Mid-turn speech no step read, or the engine refused, goes in as a prompt.
    const unread = v.isDeliveryRead ? [] : v.delivered
    v.pending = [...v.held, ...unread, ...v.pending]
    v.delivered = []
    v.held = []
    if (await read($, isActive)) {
      modLog($, `turn complete ${e.turnId} (${e.isAborted ? 'aborted' : 'answered'}, ${Math.round(e.durationMs / 1000)}s)`)
      if ((await read($, phase)) === 'thinking') await update($, phase, () => 'listening')
      scheduleSubmit($)
    }
    return result
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (e.props.hasSurvey || !(await read($, isActive))) return next(e)
    const { Box, Button, Text } = $.ui.resolve(e)
    const now = await read($, phase)
    const last = await read($, heard)
    const look = PHASE_LOOK[now]
    const glyph = now === 'thinking' ? SPINNER[(await read($, tick)) % SPINNER.length] : look.glyph
    const room = Math.max(10, e.props.bodyColumns - 50)
    const quote = last.length > room ? `${last.slice(0, room - 1)}…` : last
    const hint = now === 'thinking' && quote ? `"${quote}" · speak to add to this turn` : look.hint

    return (
      <Box>
        <Text color={look.color} bold>{`${glyph} ${now.padEnd(9)}`}</Text>
        <Text dimColor>{`  ${hint}  `}</Text>
        <Button key="voice-stop" label="Stop voice" plain onPress={() => void stop($)} />
      </Box>
    )
  })
}
