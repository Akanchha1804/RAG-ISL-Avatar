import { describe, expect, it } from 'vitest';

import {
  animatedClips,
  clipSwitchJump,
  expressionTargetsFor,
  signedCoverage,
} from './playback';

// MediaPipe hand with 21 landmarks; only the wrist (index 0) is read here.
function hand(x, y = 0.5) {
  return {
    landmarks: Array.from({ length: 21 }, (_, i) => ({ x: x + i * 0.001, y, z: 0 })),
  };
}

function frame(...hands) {
  return { hands };
}

describe('expressionTargetsFor', () => {
  it('raises the brows for a WH gloss', () => {
    expect(expressionTargetsFor(['WHAT', 'YOUR', 'NAME'])).toEqual({
      surprised: 0.55,
      angry: 0,
    });
  });

  it('furrows the brows for negation', () => {
    expect(expressionTargetsFor(['I', 'NOT', 'GO'])).toEqual({
      surprised: 0,
      angry: 0.35,
    });
  });

  it('is case-insensitive and tolerates a plain string', () => {
    expect(expressionTargetsFor('what').surprised).toBe(0.55);
  });

  it('releases both markers when there is nothing to mark', () => {
    expect(expressionTargetsFor(['HELLO', 'MORNING'])).toEqual({
      surprised: 0,
      angry: 0,
    });
    expect(expressionTargetsFor(null)).toEqual({ surprised: 0, angry: 0 });
  });
});

describe('clipSwitchJump', () => {
  const lastWrist = {
    left: { x: 0.6, y: 0.5, z: 0 },
    right: { x: 0.4, y: 0.5, z: 0 },
  };

  it('is zero when the hands do not move', () => {
    const next = frame(hand(0.4), hand(0.6));
    expect(clipSwitchJump(lastWrist, next)).toBeCloseTo(0, 6);
  });

  it('reports the worst side, not the average', () => {
    // the avatar's right hand (image-left) travels 0.3, its left stays put
    const next = frame(hand(0.1), hand(0.6));
    expect(clipSwitchJump(lastWrist, next)).toBeCloseTo(0.3, 6);
  });

  it('ignores a side with no detection instead of inventing a jump', () => {
    const next = frame(hand(0.4)); // only the avatar's right hand
    expect(clipSwitchJump({ right: { x: 0.4, y: 0.5 } }, next)).toBeCloseTo(0, 6);
  });

  it('returns zero for a frame with no usable hands', () => {
    expect(clipSwitchJump(lastWrist, frame())).toBe(0);
  });

  it('is Infinity when there is no history or no next frame', () => {
    expect(clipSwitchJump(undefined, frame(hand(0.4)))).toBe(Infinity);
    expect(clipSwitchJump(lastWrist, null)).toBe(Infinity);
  });
});

describe('animatedClips', () => {
  it('keeps only entries the avatar can play', () => {
    const animation = {
      clip_playlist: [
        { gloss: 'HELLO', landmark_clip_url: '/landmarks/gloss_hello.json' },
        { gloss: 'CAT', landmark_clip_url: null },
        { gloss: 'MORNING' },
        null,
      ],
    };
    expect(animatedClips(animation).map((c) => c.gloss)).toEqual(['HELLO']);
  });

  it('tolerates a missing animation', () => {
    expect(animatedClips(undefined)).toEqual([]);
    expect(animatedClips({})).toEqual([]);
  });
});

describe('signedCoverage', () => {
  const animation = {
    clip_playlist: [
      { gloss: 'HELLO', landmark_clip_url: '/landmarks/gloss_hello.json' },
      { gloss: 'MORNING', landmark_clip_url: null },
    ],
    unsupported_tokens: ['MORNING'],
  };

  it('counts only clips with motion on the composed path', () => {
    expect(signedCoverage(['HELLO', 'MORNING'], animation, '')).toEqual({
      composed: true,
      total: 2,
      signed: 1,
      missing: ['MORNING'],
    });
  });

  it('does not apply the counter to a sentence-level clip', () => {
    const coverage = signedCoverage(['HELLO', 'MORNING'], animation, '/landmarks/sentence.json');
    expect(coverage.composed).toBe(false);
    expect(coverage.signed).toBe(0);
  });

  it('has nothing to count without an animation', () => {
    expect(signedCoverage(['HELLO'], undefined, '')).toEqual({
      composed: false,
      total: 1,
      signed: 0,
      missing: [],
    });
  });
});
