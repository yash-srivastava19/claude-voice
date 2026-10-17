// Splits a streamed markdown reply into what is spoken and what stays on screen.
//
// Prose is spoken, sentence by sentence, as soon as each sentence is complete.
// Fenced blocks, tables, headings and rules are screen-only. The splitter is fed
// text pieces in any size, cut anywhere (mid-word, mid-fence), and returns the
// sentences that became complete.

type LineKind = 'fence' | 'table' | 'heading' | 'rule' | 'prose' | 'undecided'

const FENCE = /^\s*(```|~~~)/
const TABLE = /^\s*\|/
const HEADING = /^\s*#{1,6}(\s|$)/
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/
const MARKER = /^\s*(?:[-*+]\s+|\d+[.)]\s+|>\s?)/

// Characters that can open a non-prose line; a line starting with anything else
// is prose from its first character.
const AMBIGUOUS_START = /^\s*[`~|#>*+_\-\d]/

export function classifyLine(line: string, isComplete: boolean): LineKind {
  if (FENCE.test(line)) return 'fence'
  if (TABLE.test(line)) return 'table'
  if (HEADING.test(line)) return 'heading'
  if (isComplete && RULE.test(line)) return 'rule'
  if (line.trim() === '' && !isComplete) return 'undecided'
  if (!AMBIGUOUS_START.test(line)) return 'prose'
  // A list item or quote: prose once its marker is whole and text follows it.
  if (MARKER.test(line) && line.replace(MARKER, '').trim() !== '') {
    if (!isComplete && /^\s*[-*_]/.test(line) && RULE.test(line)) return 'undecided'
    return 'prose'
  }
  return isComplete ? 'prose' : 'undecided'
}

// Makes one sentence of markdown sound right when read aloud.
export function toSpoken(text: string): string {
  return text
    .replace(MARKER, '')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/https?:\/\/\S+/g, 'the link')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/(\*\*|__)(.+?)\1/g, '$2')
    .replace(/(^|[\s(])[*_]([^*_\s][^*_]*?)[*_](?=[\s.,;:!?)]|$)/g, '$1$2')
    .replace(/\s*(->|=>|→)\s*/g, ' to ')
    .replace(/\s+—\s+|\s+--\s+/g, ', ')
    .replace(/[*`#|<>]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

// End of a sentence: terminal punctuation, optional closing quote or paren,
// then whitespace. Skips common abbreviations so "e.g. this" stays whole.
const SENTENCE_END = /([.!?]+["')\]]?)\s+/g
const ABBREVIATION = /\b(e\.g|i\.e|etc|vs|approx|Mr|Ms|Dr|cf)\.$/i

function takeSentences(text: string): { sentences: string[]; rest: string } {
  const sentences: string[] = []
  let start = 0
  SENTENCE_END.lastIndex = 0
  for (let m = SENTENCE_END.exec(text); m; m = SENTENCE_END.exec(text)) {
    const end = m.index + (m[1] ?? "").length
    const candidate = text.slice(start, end)
    if (ABBREVIATION.test(candidate)) continue
    sentences.push(candidate)
    start = m.index + m[0].length
  }
  return { sentences, rest: text.slice(start) }
}

function speakable(sentences: string[]): string[] {
  return sentences.map(toSpoken).filter(s => /[a-z0-9]/i.test(s))
}

export class SpeechSplitter {
  private line = '' // the current, incomplete line
  private kind: LineKind = 'undecided'
  private spoken = 0 // how much of `line` already went to `prose`
  private prose = '' // prose waiting for its sentence to end
  private isInFence = false
  private block = -1

  // Feeds one streamed piece; `index` is the content block it belongs to.
  push(text: string, index = 0): string[] {
    const out: string[] = []
    if (index !== this.block) {
      out.push(...this.flush())
      this.block = index
    }
    let rest = text
    for (let nl = rest.indexOf('\n'); nl !== -1; nl = rest.indexOf('\n')) {
      this.line += rest.slice(0, nl)
      out.push(...this.endLine())
      rest = rest.slice(nl + 1)
    }
    this.line += rest
    out.push(...this.partialLine())
    return out
  }

  // Ends the stream: speaks whatever prose is left.
  flush(): string[] {
    const out = this.line ? this.endLine() : []
    const tail = this.prose
    this.prose = ''
    this.isInFence = false
    return [...out, ...speakable([tail])]
  }

  private partialLine(): string[] {
    if (this.isInFence) return []
    const out: string[] = []
    if (this.kind === 'undecided') {
      this.kind = classifyLine(this.line, false)
      const marker = this.kind === 'prose' ? this.line.match(MARKER) : null
      if (marker) {
        // A list item or quote starts its own unit, read without its marker.
        out.push(...this.breakParagraph())
        this.spoken = marker[0].length
      }
    }
    if (this.kind !== 'prose') return out
    this.prose += this.line.slice(this.spoken)
    this.spoken = this.line.length
    const { sentences, rest } = takeSentences(this.prose)
    this.prose = rest
    return [...out, ...speakable(sentences)]
  }

  private endLine(): string[] {
    const line = this.line
    const kind = this.kind === 'undecided' || this.kind === 'prose' ? classifyLine(line, true) : this.kind
    const unspoken = line.slice(this.spoken)
    const isStarted = this.spoken > 0 // part of this line is already in `prose`
    this.line = ''
    this.kind = 'undecided'
    this.spoken = 0

    if (kind === 'fence') {
      this.isInFence = !this.isInFence
      return this.breakParagraph()
    }
    if (this.isInFence || kind === 'table' || kind === 'heading' || kind === 'rule') {
      return this.breakParagraph()
    }
    if (line.trim() === '') return this.breakParagraph()

    // A list item or quote line is its own unit; a plain prose line joins the
    // paragraph, which ends at a blank line.
    const isItem = MARKER.test(line)
    const out = isItem && !isStarted ? this.breakParagraph() : []
    const text = isItem && !isStarted ? unspoken.replace(MARKER, '') : unspoken
    const separator = !isStarted && this.prose && !this.prose.endsWith(' ') ? ' ' : ''
    this.prose += separator + text
    const { sentences, rest } = takeSentences(this.prose + ' ')
    this.prose = rest.trimStart()
    out.push(...speakable(sentences))
    return isItem ? [...out, ...this.breakParagraph()] : out
  }

  private breakParagraph(): string[] {
    const tail = this.prose
    this.prose = ''
    return speakable([tail])
  }
}
