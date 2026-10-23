export type VoicePhase = 'off' | 'starting' | 'listening' | 'hearing' | 'thinking' | 'speaking' | 'asleep' | 'nomic'

declare module 'claude-code' {
  interface PluginState {
    voice: {
      isActive: boolean
      phase: VoicePhase
      heard: string
      /** Advances while thinking, to animate the spinner. */
      tick: number
    }
  }
}
