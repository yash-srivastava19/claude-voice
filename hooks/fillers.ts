// What a person says while they work, so a pause never sounds like a dropped call.
// decide() looks at the moment and returns a line to say, or nothing, and marks it in
// `said` so it isn't repeated; no other side effects. The mod
// calls it from its poll loop and speaks the line as a filler (tagged, so
// bench/turn_report.py tells it apart from the reply).

export type Moment = {
  now: number
  turnStartedAt?: number // the person's turn went in; undefined when no turn runs
  lastSpokeAt: number // anything spoken, reply or filler
  repliedThisTurn: boolean // the reply's first sentence has been sent
  tool?: { startedAt: number; name: string; description?: string }
  canSpeak: boolean // not hushed, muted, asleep, or with the person mid-sentence
  turnNumber: number // counts turns in the session, so the thinking line varies
}

// announced: the startedAt of the last tool given a line, so one call is named once.
export type Said = { thinking: boolean; toolLines: number; stillWorking: number; announced?: number }

const THINKING = ['Okay, let me think.', 'Mm, one sec.', 'Let me look.', 'Hmm.', 'Okay, give me a moment.', 'Right.']
const STILL = ['Still working on it.', 'Nearly there.', 'Still going.']

// A tool has to run this long before it is named: a quick one is over before the line
// is heard, and naming it then is stale (the 0.9 s Bash calls in the first filler drive).
const TOOL_NAMED_AFTER = 1500
const QUIET_BEFORE_TOOL_LINE = 3000
const THINKING_AFTER = 1500
const QUIET_BEFORE_STILL = 8000

// Each rule: when it applies, what it says, and how it marks the turn so it doesn't repeat.
const RULES: { when: (m: Moment, s: Said) => boolean; line: (m: Moment, s: Said) => string; mark: (s: Said, m: Moment) => void }[] = [
  {
    // The turn went in and nothing came back within 1.5 s.
    when: (m, s) => m.turnStartedAt !== undefined && !m.repliedThisTurn && !s.thinking && !m.tool && m.now - m.turnStartedAt > THINKING_AFTER,
    line: m => THINKING[m.turnNumber % THINKING.length],
    mark: s => { s.thinking = true },
  },
  {
    // A tool has run a while, unnamed, after 3 s of quiet: say once what it is doing.
    when: (m, s) => m.tool !== undefined && s.announced !== m.tool.startedAt && s.toolLines < 2
      && m.now - m.lastSpokeAt > QUIET_BEFORE_TOOL_LINE && m.now - m.tool.startedAt > TOOL_NAMED_AFTER,
    line: m => toolLine(m.tool!),
    mark: (s, m) => { s.toolLines += 1; s.announced = m.tool!.startedAt },
  },
  {
    // A long tool run: an occasional word, at most twice a turn.
    when: (m, s) => m.tool !== undefined && s.stillWorking < 2 && m.now - m.lastSpokeAt > QUIET_BEFORE_STILL
      && m.now - m.tool.startedAt > TOOL_NAMED_AFTER,
    line: (_m, s) => STILL[s.stillWorking % STILL.length],
    mark: s => { s.stillWorking += 1 },
  },
]

export function decide(m: Moment, s: Said): string | undefined {
  if (!m.canSpeak || m.turnStartedAt === undefined) return undefined
  const rule = RULES.find(r => r.when(m, s))
  if (!rule) return undefined
  const line = rule.line(m, s)
  rule.mark(s, m)
  return line
}

// Tools that carry no description, by what a person would say they are doing.
const BY_NAME: Record<string, string> = {
  WebSearch: 'Searching the web.',
  WebFetch: 'Reading the page.',
  Read: 'Reading the file.',
  Grep: 'Searching the code.',
  Glob: 'Looking for the files.',
  Edit: 'Making the change.',
  Write: 'Writing it out.',
  Agent: 'Handing part of this to a helper.',
}

export function toolLine(tool: { name: string; description?: string }): string {
  if (tool.description) return gerund(tool.description)
  return BY_NAME[tool.name] ?? 'Working on it.'
}

// "Run the test suite" -> "Running the test suite." Descriptions are imperative.
const DOUBLED = new Set(['run', 'set', 'get', 'stop', 'put', 'cut', 'plan', 'drop', 'ship', 'grep', 'diff'])

function gerund(description: string): string {
  const [verb = '', ...rest] = description.trim().replace(/[.:]+$/, '').split(/\s+/)
  const v = verb.toLowerCase()
  const ing = DOUBLED.has(v) ? `${v}${v.at(-1)}ing`
    : v.endsWith('ie') ? `${v.slice(0, -2)}ying`
    : v.endsWith('e') && !v.endsWith('ee') ? `${v.slice(0, -1)}ing`
    : `${v}ing`
  const said = [ing.charAt(0).toUpperCase() + ing.slice(1), ...rest].join(' ')
  return v ? `${said}.` : 'Working on it.'
}
