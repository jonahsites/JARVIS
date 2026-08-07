/**
 * What the glob looks like in each state.
 *
 * Your spec: blue when idle, smoothly green when it starts hearing you, yellow
 * while thinking, multicolour while responding. Every value here is tweened
 * rather than set, so nothing ever snaps — you said smoothly, not suddenly.
 *
 * Note `speaking` has tintMix: 0. That means no tint at all, which is your
 * original shader output — the full Perlin multicolour.
 */

export type JarvisState =
  | 'idle'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'offline';

export interface GlobConfig {
  perlinTime: number;
  perlinMorph: number;
  perlinDNoise: number;
  chromaRGBr: number;
  chromaRGBg: number;
  chromaRGBb: number;
  chromaRGBn: number;
  chromaRGBm: number;
  tint: [number, number, number];
  tintMix: number;
  gain: number;
  sphereWireframe: boolean;
  spherePoints: boolean;
  spherePsize: number;
  cameraSpeedY: number;
  cameraSpeedX: number;
  cameraZoom: number;
}

const BASE: GlobConfig = {
  perlinTime: 25.0,
  perlinMorph: 5.5,
  perlinDNoise: 2.5,
  chromaRGBr: 7.5,
  chromaRGBg: 5.0,
  chromaRGBb: 7.0,
  chromaRGBn: 0.0,
  chromaRGBm: 1.0,
  tint: [0.2, 0.45, 1.0],
  tintMix: 0.9,
  gain: 1.0,
  sphereWireframe: false,
  spherePoints: false,
  spherePsize: 1.0,
  cameraSpeedY: 0.08,
  cameraSpeedX: 0.0,
  cameraZoom: 175,
};

export const STATES: Record<JarvisState, GlobConfig> = {
  // Resting. Slow drift, deep blue, barely moving.
  idle: {
    ...BASE,
    perlinTime: 12.0,
    perlinMorph: 2.4,
    tint: [0.13, 0.38, 1.0],
    tintMix: 0.93,
    cameraSpeedY: 0.06,
  },

  // Hearing you. Green, and the surface starts responding — morph is driven
  // live by your mic level on top of this baseline.
  listening: {
    ...BASE,
    perlinTime: 26.0,
    perlinMorph: 6.0,
    tint: [0.15, 0.95, 0.45],
    tintMix: 0.9,
    cameraSpeedY: 0.12,
  },

  // Working. Amber, fast churn — it should read as busy without being frantic.
  thinking: {
    ...BASE,
    perlinTime: 55.0,
    perlinMorph: 9.0,
    perlinDNoise: 4.0,
    tint: [1.0, 0.76, 0.1],
    tintMix: 0.91,
    cameraSpeedY: 0.3,
  },

  // Talking. tintMix 0 = your original multicolour, unmodified.
  speaking: {
    ...BASE,
    perlinTime: 34.0,
    perlinMorph: 11.0,
    perlinDNoise: 2.0,
    chromaRGBn: 0.6,
    chromaRGBm: 2.2,
    tintMix: 0.0,
    gain: 1.9,
    cameraSpeedY: 0.16,
  },

  // Daemon not running. Nearly still, desaturated — obviously not alive.
  offline: {
    ...BASE,
    perlinTime: 4.0,
    perlinMorph: 1.0,
    tint: [0.32, 0.34, 0.4],
    tintMix: 0.97,
    cameraSpeedY: 0.02,
  },
};

/** How long each transition takes, in seconds. */
export const TRANSITION: Record<JarvisState, number> = {
  // Deliberately the slowest — this is the one you singled out as needing to
  // ease in rather than snap.
  listening: 0.85,
  thinking: 0.45,
  speaking: 0.35,
  idle: 1.1,
  offline: 1.5,
};

/**
 * Live audio reactivity layered on top of the state's baseline morph.
 * Only listening and speaking react; thinking has no audio to follow.
 */
export function morphForLevel(state: JarvisState, level: number): number {
  const base = STATES[state].perlinMorph;
  if (state === 'listening') return base + level * 14;
  if (state === 'speaking') return base + level * 9;
  return base;
}
