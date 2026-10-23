import { expect, test } from 'bun:test'

import { decide, toolLine, type Moment, type Said } from '../hooks/fillers'

const moment = (o: Partial<Moment>): Moment =>
  ({ now: 10_000, turnStartedAt: 0, lastSpokeAt: 0, repliedThisTurn: false, canSpeak: true, turnNumber: 0, ...o })
const fresh = (): Said => ({ thinking: false, toolLines: 0, stillWorking: 0 })

test.each([
  ['Run the test suite', 'Running the test suite.'],
  ['Read recent voice log lines', 'Reading recent voice log lines.'],
  ['Write live daemon benchmark', 'Writing live daemon benchmark.'],
  ['Tie up loose ends', 'Tying up loose ends.'],
  ['See which branch', 'Seeing which branch.'],
])('tool line for %p', (description, line) => {
  expect(toolLine({ name: 'Bash', description })).toBe(line)
})

test('a tool without a description is named the way a person would say it', () => {
  expect(toolLine({ name: 'WebSearch' })).toBe('Searching the web.')
  expect(toolLine({ name: 'SomethingNew' })).toBe('Working on it.')
})

test('the thinking line changes from turn to turn', () => {
  const first = decide(moment({ turnStartedAt: 8_000, turnNumber: 1 }), fresh())
  const second = decide(moment({ turnStartedAt: 8_000, turnNumber: 2 }), fresh())
  expect(first).not.toBe(second)
})

test('one tool call is named once, however long it runs', () => {
  const said = fresh()
  const tool = { startedAt: 2_000, name: 'WebSearch' }
  expect(decide(moment({ tool, lastSpokeAt: 1_000, repliedThisTurn: true }), said)).toBe('Searching the web.')
  expect(decide(moment({ now: 12_500, tool, lastSpokeAt: 9_000, repliedThisTurn: true }), said)).toBeUndefined()
})

test('a quick tool is never named: the line would land after it finished', () => {
  const tool = { startedAt: 9_000, name: 'Bash', description: 'Find the blog folder' }
  expect(decide(moment({ tool, lastSpokeAt: 1_000, repliedThisTurn: true }), fresh())).toBeUndefined()
})

test('quiet for the first 1.5 s of a turn', () => {
  expect(decide(moment({ turnStartedAt: 9_000 }), fresh())).toBeUndefined()
})

test('one thinking line once the turn is 1.5 s old with no reply', () => {
  const said = fresh()
  expect(decide(moment({ turnStartedAt: 8_000 }), said)).toBe('Okay, let me think.')
  expect(decide(moment({ turnStartedAt: 8_000 }), said)).toBeUndefined()
})

test('never over the person, outside a turn, or once the reply started', () => {
  expect(decide(moment({ canSpeak: false }), fresh())).toBeUndefined()
  expect(decide(moment({ turnStartedAt: undefined }), fresh())).toBeUndefined()
  expect(decide(moment({ repliedThisTurn: true }), fresh())).toBeUndefined()
})

test('says what a tool is doing after 3 s of quiet, not right after speaking', () => {
  const tool = { startedAt: 8_000, name: 'Bash', description: 'Run the tests' }
  expect(decide(moment({ tool, lastSpokeAt: 6_500, repliedThisTurn: true }), fresh())).toBe('Running the tests.')
  expect(decide(moment({ tool, lastSpokeAt: 8_500, repliedThisTurn: true }), fresh())).toBeUndefined()
})

test('a long tool run gets an occasional word, at most twice', () => {
  const said = { ...fresh(), toolLines: 2 }
  const long = moment({ tool: { startedAt: 0, name: 'Bash', description: 'Run' }, lastSpokeAt: 1_000, repliedThisTurn: true })
  expect(decide(long, said)).toBe('Still working on it.')
  expect(decide(long, said)).toBe('Nearly there.')
  expect(decide(long, said)).toBeUndefined()
})
