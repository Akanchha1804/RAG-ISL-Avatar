// Shared test setup: unmount rendered components between tests so one test
// cannot leak DOM into the next.
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => {
  cleanup();
});
