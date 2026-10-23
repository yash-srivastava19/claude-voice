import { expect, test } from 'bun:test'

import { decide, toolLine, type Moment, type Said } from '../hooks/fillers'

const moment = (o: Partial<Moment>): Moment =>
  ({ now: 10_000, turnStartedAt: 0, lastSpokeAt: 0, repliedThisTurn: false, canSpeak: true, ...o })
const fresh = (): Said => ({ thinking: false, toolLines: 0, stillWorking: 0 })

test.each([
  ['Run the test suite', 'Running the test suite.'],
  ['Read recent voice log lines', 'Reading recent voice log lines.'],
  ['Write live daemon benchmark', 'Writing live daemon benchmark.'],
  ['Tie up loose ends', 'Tying up loose ends.'],
  ['See which branch', 'Seeing which branch.'],
])('tool line for %p', (description, line) => {
  expect(toolLine(description)).toBe(line)
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

test('says what a tool is doing after 2 s of quiet, not right after speaking', () => {
  const tool = { startedAt: 9_000, description: 'Run the tests' }
  expect(decide(moment({ tool, lastSpokeAt: 7_000, repliedThisTurn: true }), fresh())).toBe('Running the tests.')
  expect(decide(moment({ tool, lastSpokeAt: 9_500, repliedThisTurn: true }), fresh())).toBeUndefined()
})

test('a long tool run gets an occasional word, at most twice', () => {
  const said = { ...fresh(), toolLines: 3 }
  const long = moment({ tool: { startedAt: 0, description: 'Run' }, lastSpokeAt: 1_000, repliedThisTurn: true })
  expect(decide(long, said)).toBe('Still working on it.')
  expect(decide(long, said)).toBe('Nearly there.')
  expect(decide(long, said)).toBeUndefined()
})
