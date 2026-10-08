import { wsUrl } from './config';

const MAX_BACKOFF_MS = 15000;
const INITIAL_BACKOFF_MS = 1000;

let socket = null;
let reconnectTimer = null;
let backoff = INITIAL_BACKOFF_MS;

// True while a consumer still wants the socket up. Replaces the old
// `intentionalClose` flag: that flag was reset by the reconnecting mount
// before the previous socket's async `onclose` arrived, so a stale socket
// could clear `socket` and orphan the healthy one (React StrictMode
// double-mounts effects in dev, so this happened on every page load).
let wantConnection = false;

let messageHandler = null;
let errorHandler = null;
let openHandler = null;
let closeHandler = null;

function clearReconnectTimer() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function scheduleReconnect() {
  clearReconnectTimer();
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    const delay = backoff;
    backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
    console.log(`[WS] Reconnecting in ${delay}ms...`);
    ensureSocket();
  }, delay);
}

function createSocket() {
  const url = wsUrl('/api/pipeline/ws');
  const sock = new WebSocket(url);
  socket = sock;

  // Every handler is a no-op unless this socket is still the current one,
  // so a late event from a replaced socket cannot corrupt live state.
  sock.onopen = () => {
    if (socket !== sock) return;
    backoff = INITIAL_BACKOFF_MS;
    clearReconnectTimer();
    console.log('[WS] Connected to pipeline');
    if (openHandler) openHandler();
  };

  sock.onmessage = (event) => {
    if (socket !== sock) return;
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      console.error('[WS] Parse error:', e);
      return;
    }
    if (messageHandler) messageHandler(data);
  };

  sock.onerror = () => {
    if (socket !== sock) return;
    // A WebSocket error event carries no message; the close event that
    // follows has the code. Report something actionable instead of the
    // raw event object.
    const detail = `connection to ${url} failed (readyState=${sock.readyState})`;
    console.error(`[WS] ${detail}`);
    if (errorHandler) errorHandler(new Error(detail));
  };

  sock.onclose = (event) => {
    if (socket !== sock) return;
    socket = null;
    console.log(
      `[WS] Disconnected code=${event.code}${event.reason ? ` reason="${event.reason}"` : ''}`
    );
    if (closeHandler) closeHandler(event.code);
    if (!wantConnection) return;
    scheduleReconnect();
  };

  return sock;
}

function ensureSocket() {
  if (!wantConnection) return socket;
  if (
    socket &&
    (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)
  ) {
    return socket;
  }
  return createSocket();
}

export function connectPipeline({ onMessage, onError, onOpen, onClose } = {}) {
  if (onMessage) messageHandler = onMessage;
  if (onError) errorHandler = onError;
  if (onOpen) openHandler = onOpen;
  if (onClose) closeHandler = onClose;

  wantConnection = true;
  return ensureSocket();
}

export function sendPipelineMessage(payload) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    throw new Error('WebSocket not connected');
  }
  socket.send(JSON.stringify(payload));
}

export function sendTextForTranslation(text, useRag = true, topK = 5) {
  sendPipelineMessage({
    input_mode: 'text',
    text: text,
    use_rag: useRag,
    top_k: topK,
  });
}

export function sendAudioForTranslation(audioBase64, useRag = true, topK = 5, language = null) {
  sendPipelineMessage({
    input_mode: 'speech',
    audio_base64: audioBase64,
    use_rag: useRag,
    top_k: topK,
    ...(language ? { language } : {}),
  });
}

export function disconnectPipeline() {
  wantConnection = false;
  clearReconnectTimer();
  backoff = INITIAL_BACKOFF_MS;

  const sock = socket;
  socket = null;
  if (!sock) return;

  // Detach first so the close event cannot re-enter our handlers.
  sock.onopen = null;
  sock.onmessage = null;
  sock.onerror = null;
  sock.onclose = null;
  try {
    sock.close(1000);
  } catch (e) {
    console.warn('[WS] close failed:', e);
  }
}

export function isConnected() {
  return !!socket && socket.readyState === WebSocket.OPEN;
}