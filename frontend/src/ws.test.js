import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { connectPipeline, disconnectPipeline, isConnected } from './ws';

// Minimal WebSocket stand-in: records every connection and lets a test drive
// open/close synchronously, which is all the reconnect logic needs.
class FakeSocket {
  static CONNECTING = 0;

  static OPEN = 1;

  static CLOSING = 2;

  static CLOSED = 3;

  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = FakeSocket.CONNECTING;
    FakeSocket.instances.push(this);
  }

  close() {
    this.readyState = FakeSocket.CLOSED;
  }

  send() {}

  open() {
    this.readyState = FakeSocket.OPEN;
    if (this.onopen) this.onopen();
  }

  drop(code = 1006) {
    this.readyState = FakeSocket.CLOSED;
    if (this.onclose) this.onclose({ code });
  }
}

const last = () => FakeSocket.instances[FakeSocket.instances.length - 1];

describe('websocket reconnect', () => {
  beforeEach(() => {
    FakeSocket.instances = [];
    vi.useFakeTimers();
    vi.stubGlobal('WebSocket', FakeSocket);
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    disconnectPipeline();
  });

  afterEach(() => {
    disconnectPipeline();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('reconnects one second after the socket drops', () => {
    connectPipeline({});
    expect(FakeSocket.instances).toHaveLength(1);

    FakeSocket.instances[0].open();
    expect(isConnected()).toBe(true);

    FakeSocket.instances[0].drop();
    // The close handler used to throw "delay is not defined" here, so no
    // reconnect was ever scheduled.
    vi.advanceTimersByTime(999);
    expect(FakeSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeSocket.instances).toHaveLength(2);
  });

  it('backs off exponentially and caps at 15 seconds', () => {
    connectPipeline({});
    const waits = [1000, 2000, 4000, 8000, 15000, 15000];
    waits.forEach((wait) => {
      last().drop();
      const before = FakeSocket.instances.length;
      vi.advanceTimersByTime(wait - 1);
      expect(FakeSocket.instances).toHaveLength(before);
      vi.advanceTimersByTime(1);
      expect(FakeSocket.instances).toHaveLength(before + 1);
    });
  });

  it('resets the backoff after a successful connection', () => {
    connectPipeline({});
    FakeSocket.instances[0].drop();
    vi.advanceTimersByTime(1000);

    last().open();
    last().drop();
    const before = FakeSocket.instances.length;
    vi.advanceTimersByTime(1000);
    expect(FakeSocket.instances).toHaveLength(before + 1);
  });

  it('stops reconnecting after an intentional disconnect', () => {
    connectPipeline({});
    FakeSocket.instances[0].drop();
    disconnectPipeline();
    vi.advanceTimersByTime(60000);
    expect(FakeSocket.instances).toHaveLength(1);
  });
});
