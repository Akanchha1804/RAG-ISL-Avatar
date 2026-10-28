import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import GlossDisplay from './GlossDisplay';

function result(overrides = {}) {
  return {
    input_text: 'hello morning',
    glosses: 'HELLO MORNING',
    gloss_sequence: ['HELLO', 'MORNING'],
    method: 'word_by_word',
    landmark_url: '',
    landmark_file: '',
    animation: {
      clip_playlist: [],
      unresolved_tokens: [],
      unsupported_tokens: ['MORNING'],
      retrieval_detail: { level: 'composed', store: 'json' },
    },
    ...overrides,
  };
}

describe('GlossDisplay', () => {
  it('states that this build has no sign animation', () => {
    render(<GlossDisplay result={result()} />);
    expect(screen.getByText('No sign animation in this build: glosses only.')).toBeTruthy();
    expect(screen.queryByText(/Playing .* gloss clips/)).toBeNull();
  });

  it('names tokens that are not in the vocabulary', () => {
    render(
      <GlossDisplay
        result={result({
          gloss_sequence: ['HELLO', 'Rahul'],
          glosses: 'HELLO Rahul',
          animation: {
            clip_playlist: [],
            unresolved_tokens: ['Rahul'],
            unsupported_tokens: ['Rahul'],
          },
        })}
      />,
    );
    expect(screen.getByText('Not in vocabulary: Rahul')).toBeTruthy();
  });

  it('shows the English conversion for Hindi input', () => {
    render(
      <GlossDisplay
        result={result({
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
        result={result({
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
