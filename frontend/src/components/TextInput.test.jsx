import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import TextInput from './TextInput';

describe('TextInput demo chips', () => {
  it('translates a demo sentence in one click, with RAG on by default', () => {
    const onTranslate = vi.fn();
    render(<TextInput onTranslate={onTranslate} loading={false} />);

    fireEvent.click(screen.getByRole('button', { name: 'help me' }));

    expect(onTranslate).toHaveBeenCalledWith('help me', true);
  });

  it('offers a Hindi example that exercises the translation stage', () => {
    render(<TextInput onTranslate={vi.fn()} loading={false} />);

    expect(screen.getByRole('button', { name: 'सुप्रभात' })).toBeTruthy();
  });

  it('does not queue another run while a request is in flight', () => {
    const onTranslate = vi.fn();
    render(<TextInput onTranslate={onTranslate} loading />);

    fireEvent.click(screen.getByRole('button', { name: 'help me' }));

    expect(onTranslate).not.toHaveBeenCalled();
  });

  it('trims typed text and respects the RAG checkbox', () => {
    const onTranslate = vi.fn();
    render(<TextInput onTranslate={onTranslate} loading={false} />);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: '  good morning  ' } });
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: 'Translate to ISL' }));

    expect(onTranslate).toHaveBeenCalledWith('good morning', false);
  });
});
