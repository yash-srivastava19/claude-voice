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
  nomic: { glyph: '✕', color: 'red', hint: 'mic lost · reconnecting' },
}
const SPINNER = ['◐', '◓', '◑', '◒']
const POLL_MS = 250
const KEEP_AWAKE_MS = 20_000
// Once no speech segment is open, wait this long before submitting. Segments
// still being transcribed hold the prompt open on their own (see openSegments).
const SUBMIT_DEBOUNCE_MS = 400
const STOP_WORDS = /^(stop|wait|hold on|cancel|never mind|shut up)[.!]?$/i
const VOICES_DIR = '~/.local/share/voice-mode/voices'
const HELP = `/talk start, /talk stop     turn voice mode on or off
/talk hush                  stop the reply being spoken; voice mode stays on
/talk voices                list the voices
/talk voice <name>          switch voice: a name from the list, a mix such as
                            af_heart:0.6+bf_emma:0.4, or one of your own
/talk speed <0.5 to 2.0>    speaking rate (1 is normal)
/talk settings              show the current voice and speed`

const VOICE_CONTEXT = `Voice mode is on. The person is talking to you out loud through speech recognition, and every prose sentence you write is read aloud to them by a text-to-speech voice. Fenced code blocks, tables and headings are shown on screen and are not read. Talk the way a good pair programmer does: you are both looking at the same screen.

What to say and what to show:
- Say: the point, the cause, the decision, one or two key numbers, what you are doing now, and a question back.
- Show, in a fenced code block or a table: links and sources, file paths, commands, code, identifiers, full sets of numbers, long lists, and anything they would copy.
- The test: if they need to see it to use it, or it is more than about two items, it goes on screen.
- Sources and links go in a table at the end. Say "sources are on screen" and never read them.
- Prose outside a fence or table is spoken, lists included. Keep spoken lists to two or three short phrases; move longer ones into a table.

How to talk:
- Explain at the middle level: how the pieces connect and why, not line by line and not only the goal.
- Say the substance of what is on screen; the screen backs up what you said. Never just point ("have a look at the table") without saying what it shows.
- For a diagram, walk it in order, name the part you are on, and stop on the part that matters.
- One idea per turn, then hand the turn back, for example by asking whether to go on.
- While working, one short line per real step ("opening the log now"). No filler.
- Short, natural spoken sentences, first point first. Go deeper only when asked.
- Transcripts may mishear technical words; infer the meaning, and ask when unsure.

Examples of the style:
- Instead of "Have a look at the table": "The slow part is speech to text, almost four seconds, when it is usually under one. Funes has most of the cores. The numbers are on screen."
- Walking a diagram: "First check is easy: if I am talking, the mic is paused. Then it waits half a second, because the speakers ring on. That middle box is the one we fixed. After that it just listens."
- Progress: "Opening the voice log." Then later: "Right, found it."
- Handing back: "That is the cause. Want me to fix it, or look at the numbers first?"
- Sources: "That is from OpenAI's voice guide and a pair programming study. Sources are on screen."`

// Per-load state: a reload evaluates the module afresh, and `session.start`
// resumes from $.state.
const v = {
  socket: '',
  // How the daemon keeps its own voice out of the mic: aec, gate or open (see voiced.py).
  echo: '',
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
  // Set by Hush: the turn keeps working, but says nothing more until it ends.
  isHushedTurn: false,
  // Speech delivered into the running turn, and whether a step has started since
  // (a step that starts after the delivery is the one that reads it).
  delivered: [] as string[],
  isDeliveryRead: false,
  // Mid-turn speech the engine refused to append; it goes in when the turn ends.
  held: [] as string[],
  hasSpokenThisTurn: false,
  // The daemon sleeps after a quiet spell; a running turn keeps it awake.
  lastKeepAwake: 0,
  // The plugin's settings (userConfig), from `register`'s options; /talk voice and
  // /talk speed change them through /config, which reloads the module with the new ones.
  settings: { voice: 'michael', speed: 1.05 },
}

async function call($: EngineInterface, method: 'GET' | 'POST', path: string, body?: object) {
  const res = await $.http.fetch(`http://voice${path}`, {
    method,
    socketPath: v.socket,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const why = (() => { try { return JSON.parse(res.text).error } catch { return undefined } })()
    throw new Error(why ?? `voice daemon ${path}: HTTP ${res.status}`)
  }
  return JSON.parse(res.text || '{}')
}

// First /talk start: install the Python environment and models (install.sh is
// idempotent and leaves a marker when done).
async function ensureInstalled($: EngineInterface) {
  const check = await $.process.run(['sh', `${$.plugin.root}/install.sh`, '--check'])
  if (check.exitCode === 0) return
  $.ui.toast('Voice: setting up (first run or after an update; the first run downloads about 500 MB).', { timeoutMs: 20_000 })
  // Downloads resume, so a run cut off by the cap picks up where it stopped next time.
  const { exitCode, stderr } = await $.process.run(['sh', `${$.plugin.root}/install.sh`], { timeoutMs: 1_800_000 })
  if (exitCode !== 0) {
    // install.sh explains a failure from its "voice: " line on, with the last lines of
    // the tool that failed (uv's cause sits mid-chain, above "operation timed out").
    const lines = stderr.trim().split('\n').filter(Boolean)
    const at = lines.findIndex(l => l.startsWith('voice: '))
    const why = (at >= 0 ? lines.slice(at).join('\n').slice('voice: '.length) : lines.slice(-8).join('\n')) || 'setup failed'
    throw new Error(`${why}\nFull log: ~/.local/share/voice-mode/install.log. Run sh ${$.plugin.root}/install.sh to retry.`)
  }
}

// Runs the daemon script with the voice environment's Python.
function daemonScript($: EngineInterface, args: string[], timeoutMs = 30_000) {
  return $.process.run(
    ['sh', '-c', '"${VOICE_PYTHON:-$HOME/.local/share/voice-mode/venv/bin/python}" "$@"', 'sh',
      `${$.plugin.root}/helper/voiced.py`, ...args],
    { timeoutMs },
  )
}

async function startDaemon($: EngineInterface) {
  // --start also replaces a daemon left running by an older version of the plugin.
  const { exitCode, stdout, stderr } = await daemonScript($, ['--start'])
  if (exitCode !== 0) throw new Error(stderr.trim() || `voice daemon exited ${exitCode}`)
  v.socket = JSON.parse(stdout.trim().split('\n').pop() ?? '{}').socket
  v.echo = (await call($, 'GET', '/health').catch(() => ({}))).echo ?? ''
  await call($, 'POST', '/config', v.settings).catch(err =>
    $.ui.toast(`voice: ${(err as Error).message}; keeping the previous voice`))
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

// Hush: stop speaking now and stay quiet for the rest of the turn, which keeps working.
function hushTurn($: EngineInterface, why: string) {
  hush($, why)
  if (v.runningTurn) v.isHushedTurn = true
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
      } else if (ev.kind === 'mic') {
        const ok = (ev as { ok?: boolean }).ok
        modLog($, ok ? 'mic back' : 'mic lost; reconnecting')
        await update($, phase, () => (!ok ? 'nomic' : v.runningTurn ? 'thinking' : 'listening'))
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

// Saves one of the plugin's settings the way the /config menu would.
async function saveSetting($: EngineInterface, field: 'voice' | 'speed', value: string | number) {
  const row = (await $.config.list()).find(r => /^voice(@[^.]*)?\./.test(r.key) && r.key.endsWith(`.${field}`))
  if (!row) return 'saved for this session only (no /config row found)'
  const { deny } = await $.config.set({ key: row.key, value })
  return deny ? `not saved: ${deny}` : undefined
}

async function listVoices($: EngineInterface) {
  const { exitCode, stdout, stderr } = await daemonScript($, ['--voices'])
  if (exitCode !== 0) return `Could not list voices: ${stderr.trim()}`
  const all = JSON.parse(stdout) as {
    presets: { name: string; about: string }[]
    kokoro: string[]
    yours: { name: string }[]
  }
  const width = Math.max(...all.presets.map(p => p.name.length)) + 2
  return [
    `Current: ${v.settings.voice} at speed ${v.settings.speed}. Switch with /talk voice <name>.`,
    '',
    ...all.presets.map(p => `  ${p.name.padEnd(width)}${p.about}`),
    '',
    `All Kokoro voices (a = US, b = British; f/m): ${all.kokoro.join(', ')}`,
    'Mix them with shares: /talk voice af_heart:0.6+bf_emma:0.4',
    '',
    all.yours.length
      ? `Yours: ${all.yours.map(y => y.name).join(', ')}`
      : `Yours: none yet. Put a voice file in ${VOICES_DIR} (see the README).`,
  ].join('\n')
}

// Cuts off whatever is playing, then speaks a sample. The flush is awaited: sent
// unawaited, it could reach the daemon after the sample and drop it.
async function preview($: EngineInterface, why: string, text: string) {
  v.gen += 1
  await call($, 'POST', '/flush', { gen: v.gen, why })
  say($, text)
}

async function setVoice($: EngineInterface, name: string) {
  const check = await daemonScript($, ['--check-voice', name])
  if (check.exitCode !== 0) return check.stderr.trim() || `No voice named ${name}.`
  v.settings = { ...v.settings, voice: name }
  if (await read($, isActive)) {
    await call($, 'POST', '/config', { voice: name })
    await preview($, 'voice change', `Hi, this is ${name}. This is how I sound.`)
  }
  const note = await saveSetting($, 'voice', name)
  return `Voice: ${name}${note ? ` (${note})` : ''}.`
}

async function setSpeed($: EngineInterface, arg: string) {
  const speed = Number(arg)
  if (!arg || !Number.isFinite(speed) || speed < 0.5 || speed > 2) return 'Speed is a number from 0.5 to 2.0, such as /talk speed 1.1.'
  v.settings = { ...v.settings, speed }
  if (await read($, isActive)) {
    await call($, 'POST', '/config', { speed })
    await preview($, 'speed change', 'This is how fast I talk now.')
  }
  const note = await saveSetting($, 'speed', speed)
  return `Speed: ${speed}${note ? ` (${note})` : ''}.`
}

export const register: Register = (on, options) => {
  if (typeof options.voice === 'string' && options.voice.trim()) v.settings.voice = options.voice.trim()
  if (typeof options.speed === 'number') v.settings.speed = options.speed

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
        description: 'Talk to Claude hands-free: /talk start, /talk stop, /talk voices',
        argumentHint: '[start|stop|voices|voice <name>|speed <n>|settings]',
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
    const [sub = '', ...rest] = e.args.trim().split(/\s+/)
    const arg = sub.toLowerCase()
    const value = rest.join(' ')
    try {
      if (arg === 'voices') return { text: await listVoices($) }
      if (arg === 'voice') return { text: value ? await setVoice($, value) : `Voice: ${v.settings.voice}. /talk voices lists the others.` }
      if (arg === 'speed') return { text: value ? await setSpeed($, value) : `Speed: ${v.settings.speed}.` }
      if (arg === 'hush') {
        hushTurn($, 'hush command')
        return { text: 'Hushed.' }
      }
      if (arg === 'settings') return { text: `Voice: ${v.settings.voice}\nSpeed: ${v.settings.speed}\nYour voices folder: ${VOICES_DIR}` }
    } catch (err) {
      return { text: `voice: ${(err as Error).message}` }
    }
    if (!['', 'start', 'on', 'stop', 'off'].includes(arg)) return { text: HELP }
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
    if (v.echo === 'gate') {
      return { text: 'Voice mode on. Talk whenever you are ready. No echo cancellation here, so the mic pauses while I speak. To interrupt, press Hush, run /talk hush, or send a message; with headphones, set VOICE_ECHO=open to talk over me. /talk stop to end.' }
    }
    return { text: 'Voice mode on. Talk whenever you are ready; say "stop" to interrupt. /talk stop to end.' }
  })

  on('prompt.submit', async ($, e, next) => {
    if (!(await read($, isActive))) return next(e)
    // A prompt from the person, typed or spoken, ends any reply still being read. One
    // that only arrives (a background task finishing, a scheduled check-in) waits its
    // turn: its answer is spoken after the current reply.
    const o = e.origin
    if (o.kind === 'composer' || o.kind === 'bridge' || (o.kind === 'plugin' && o.asUser)) hush($, 'new prompt')
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
      if (v.muteUntilNextStep || v.isHushedTurn) return
      for (const s of sentences) say($, s)
    }
    const stream = next(e)
    for await (const chunk of stream) {
      yield chunk
      if (chunk.kind === 'text') speak(splitter.push(chunk.text, chunk.index))
    }
    speak(splitter.flush())
  })

  // Tool runs are where a turn goes quiet; the log marks them so the gaps can be measured.
  on('tool.call', async ($, e, next) => {
    if (e.agentId || !(await read($, isActive))) return next(e)
    const what = 'description' in e && typeof e.description === 'string' ? `: ${e.description}` : ''
    modLog($, `tool start ${e.tool}${what}`)
    const started = await $.clock.now()
    try {
      return await next(e)
    } finally {
      modLog($, `tool end ${e.tool} (${((await $.clock.now()) - started) / 1000}s)`)
    }
  })

  on('turn.complete', async ($, e, next) => {
    const result = await next(e)
    if (e.agentId) return result
    v.runningTurn = undefined
    v.muteUntilNextStep = false
    v.isHushedTurn = false
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
    const hint = now === 'thinking' && quote ? `"${quote}" · speak to add to this turn`
      : now === 'speaking' && v.echo === 'gate' ? 'mic paused while I speak · Hush or send a message to interrupt'
      : look.hint

    return (
      <Box>
        <Text color={look.color} bold>{`${glyph} ${now.padEnd(9)}`}</Text>
        <Text dimColor>{`  ${hint}  `}</Text>
        {now === 'speaking' && <Button key="voice-hush" label="Hush" plain onPress={() => hushTurn($, 'hush button')} />}
        <Button key="voice-stop" label="Stop voice" plain onPress={() => void stop($)} />
      </Box>
    )
  })
}
