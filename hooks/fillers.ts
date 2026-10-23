// What a person says while they work, so a pause never sounds like a dropped call.
// Pure: decide() looks at the moment and returns a line to say, or nothing. The mod
// calls it from its poll loop and speaks the line as a filler (tagged, so
// bench/turn_report.py tells it apart from the reply).

export type Moment = {
  now: number
  turnStartedAt?: number // the person's turn went in; undefined when no turn runs
  lastSpokeAt: number // anything spoken, reply or filler
  repliedThisTurn: boolean // the reply's first sentence has been sent
  tool?: { startedAt: number; description: string }
  canSpeak: boolean // not hushed, muted, asleep, or with the person mid-sentence
}

export type Said = { thinking: boolean; toolLines: number; stillWorking: number }

const THINKING = ['Okay, let me think.', 'Mm, one sec.', 'Let me look.', 'Okay, give me a moment.']
const STILL = ['Still working on it.', 'Nearly there.', 'Still going.']

// Each rule: when it applies, what it says, and how it marks the turn so it doesn't repeat.
const RULES: { when: (m: Moment, s: Said) => boolean; line: (m: Moment, s: Said) => string; mark: (s: Said) => void }[] = [
  {
    // The turn went in and nothing came back within 1.5 s.
    when: (m, s) => m.turnStartedAt !== undefined && !m.repliedThisTurn && !s.thinking && !m.tool && m.now - m.turnStartedAt > 1500,
    line: (_m, s) => THINKING[(s.toolLines + s.stillWorking) % THINKING.length],
    mark: s => { s.thinking = true },
  },
  {
    // A tool started and nothing was said in the last 2 s: say what it is doing.
    when: (m, s) => m.tool !== undefined && s.toolLines < 3 && m.now - m.lastSpokeAt > 2000 && m.now - m.tool.startedAt > 300,
    line: m => toolLine(m.tool!.description),
    mark: s => { s.toolLines += 1 },
  },
  {
    // A long tool run: an occasional word, at most twice a turn.
    when: (m, s) => m.tool !== undefined && s.stillWorking < 2 && m.now - m.lastSpokeAt > 8000,
    line: (_m, s) => STILL[s.stillWorking % STILL.length],
    mark: s => { s.stillWorking += 1 },
  },
]

export function decide(m: Moment, s: Said): string | undefined {
  if (!m.canSpeak || m.turnStartedAt === undefined) return undefined
  const rule = RULES.find(r => r.when(m, s))
  if (!rule) return undefined
  const line = rule.line(m, s)
  rule.mark(s)
  return line
}

// "Run the test suite" -> "Running the test suite." Descriptions are imperative.
const DOUBLED = new Set(['run', 'set', 'get', 'stop', 'put', 'cut', 'plan', 'drop', 'ship', 'grep', 'diff'])

export function toolLine(description: string): string {
  const [verb = '', ...rest] = description.trim().replace(/[.:]+$/, '').split(/\s+/)
  const v = verb.toLowerCase()
  const ing = DOUBLED.has(v) ? `${v}${v.at(-1)}ing`
    : v.endsWith('ie') ? `${v.slice(0, -2)}ying`
    : v.endsWith('e') && !v.endsWith('ee') ? `${v.slice(0, -1)}ing`
    : `${v}ing`
  const said = [ing.charAt(0).toUpperCase() + ing.slice(1), ...rest].join(' ')
  return v ? `${said}.` : 'Working on it.'
}
