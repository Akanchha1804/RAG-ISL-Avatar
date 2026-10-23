import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import GlossDisplay from './GlossDisplay';

const playable = (gloss) => ({
  gloss,
  landmark_clip_url: `/landmarks/${gloss.toLowerCase()}.json`,
});

function composedResult(overrides = {}) {
  return {
    input_text: 'hello morning',
    glosses: 'HELLO MORNING',
    gloss_sequence: ['HELLO', 'MORNING'],
    method: 'word_by_word',
    landmark_url: '',
    landmark_file: '',
    animation: {
      clip_playlist: [playable('HELLO'), { gloss: 'MORNING', landmark_clip_url: null }],
      unresolved_tokens: [],
      unsupported_tokens: ['MORNING'],
      retrieval_detail: { level: 'composed', store: 'json' },
    },
    ...overrides,
  };
}

describe('GlossDisplay coverage', () => {
  it('says how many glosses are signed and names the rest', () => {
    render(<GlossDisplay result={composedResult()} />);
    expect(screen.getByText('Signed 1 of 2 glosses')).toBeTruthy();
    expect(screen.getByText('Not signed (no motion yet): MORNING')).toBeTruthy();
  });

  it('says all glosses are signed when every clip has motion', () => {
    render(
      <GlossDisplay
        result={composedResult({
          animation: {
            clip_playlist: [playable('HELLO'), playable('MORNING')],
            unresolved_tokens: [],
            unsupported_tokens: [],
          },
        })}
      />,
    );
    expect(screen.getByText('Signed all 2 glosses')).toBeTruthy();
    expect(screen.queryByText(/Not signed/)).toBeNull();
  });

  it('does not apply the composed counter to a sentence-level clip', () => {
    render(
      <GlossDisplay
        result={composedResult({
          landmark_url: '/landmarks/hello_morning.json',
          landmark_file: 'hello_morning.json',
          animation: { clip_playlist: [playable('HELLO')], unresolved_tokens: [] },
        })}
      />,
    );
    expect(screen.queryByText(/^Signed /)).toBeNull();
  });

  it('names tokens that are not in the vocabulary', () => {
    render(
      <GlossDisplay
        result={composedResult({
          gloss_sequence: ['HELLO', 'Rahul'],
          glosses: 'HELLO Rahul',
          animation: {
            clip_playlist: [playable('HELLO')],
            unresolved_tokens: ['Rahul'],
            unsupported_tokens: ['Rahul'],
          },
        })}
      />,
    );
    expect(screen.getByText(/Signed 1 of 2 glosses/)).toBeTruthy();
    expect(screen.getByText('Not in vocabulary: Rahul')).toBeTruthy();
  });
});
