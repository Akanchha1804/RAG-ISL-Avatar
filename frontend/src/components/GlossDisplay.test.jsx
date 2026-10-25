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
  it('says how many gloss clips play and names the rest', () => {
    render(<GlossDisplay result={composedResult()} />);
    expect(screen.getByText('Playing 1 of 2 gloss clips')).toBeTruthy();
    expect(screen.getByText('Not signed (no motion yet): MORNING')).toBeTruthy();
  });

  it('says all gloss clips play when every clip has motion', () => {
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
    expect(screen.getByText('Playing all 2 gloss clips')).toBeTruthy();
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
    expect(screen.queryByText(/^Playing /)).toBeNull();
    expect(screen.getByText(/Recorded sentence clip/)).toBeTruthy();
  });

  it('labels composed playback as experimental, not fluent ISL', () => {
    render(<GlossDisplay result={composedResult()} />);
    expect(screen.getByText(/Experimental gloss playback/)).toBeTruthy();
    expect(screen.getByText(/not a fluent ISL translation/)).toBeTruthy();
  });

  it('explains when the recorded source uses one hand', () => {
    render(
      <GlossDisplay
        result={composedResult()}
        sourceInfo={{ mode: 'composed', clips: 2, oneHandedClips: 2, oneHanded: true }}
      />,
    );
    expect(
      screen.getByText('2 of 2 source clips are largely one-handed; the other hand rests or holds.'),
    ).toBeTruthy();
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
    expect(screen.getByText(/Playing 1 of 2 gloss clips/)).toBeTruthy();
    expect(screen.getByText('Not in vocabulary: Rahul')).toBeTruthy();
  });

  it('shows the English conversion for Hindi input', () => {
    render(
      <GlossDisplay
        result={composedResult({
          input_text: 'सुप्रभात',
          gloss_sequence: ['GOOD', 'MORNING'],
          glosses: 'GOOD MORNING',
          translation: {
            detected_language: 'hi',
            source_text: 'सुप्रभात',
            translated_text: 'Good morning',
          },
        })}
      />,
    );
    expect(screen.getByText('Translated (hi → en)')).toBeTruthy();
    expect(screen.getByText('Good morning')).toBeTruthy();
  });

  it('shows why Hindi input was not translated', () => {
    render(
      <GlossDisplay
        result={composedResult({
          input_text: 'सुप्रभात',
          translation: {
            detected_language: 'hi',
            source_text: 'सुप्रभात',
            translated_text: null,
            warning: 'translation unavailable; original text used',
          },
        })}
      />,
    );
    expect(screen.getByText('translation unavailable; original text used')).toBeTruthy();
  });
});
