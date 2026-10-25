import { describe, expect, it } from 'vitest';

import { resolveUrl, withBase } from './config';

describe('withBase', () => {
  it('moves site-relative asset paths onto the API host', () => {
    expect(withBase('/landmarks/gloss_hello.json', 'https://api.example')).toBe(
      'https://api.example/landmarks/gloss_hello.json',
    );
  });

  it('leaves absolute and empty URLs alone', () => {
    expect(withBase('https://cdn.example/landmarks/a.json', 'https://api.example')).toBe(
      'https://cdn.example/landmarks/a.json',
    );
    expect(withBase('', 'https://api.example')).toBe('');
    expect(withBase(null, 'https://api.example')).toBe(null);
  });

  it('is a no-op for a same-origin deployment', () => {
    expect(resolveUrl('/avatar/model.vrm')).toBe('/avatar/model.vrm');
  });
});
