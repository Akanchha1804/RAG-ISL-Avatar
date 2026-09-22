const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE = API_BASE.replace(/^http/, 'ws');

let socket = null;
let reconnectTimer = null;
let messageHandler = null;
let errorHandler = null;

export function connectPipeline(onMessage, onError) {
  if (socket && socket.readyState === WebSocket.OPEN) return socket;

  messageHandler = onMessage;
  errorHandler = onError;

  const wsUrl = `${WS_BASE}/api/pipeline/ws`;
  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log('[WS] Connected to pipeline');
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (messageHandler) messageHandler(data);
    } catch (e) {
      console.error('[WS] Parse error:', e);
    }
  };

  socket.onerror = (err) => {
    console.error('[WS] Error:', err);
    if (errorHandler) errorHandler(err);
  };

  socket.onclose = (event) => {
    console.log('[WS] Disconnected', event.code);
    socket = null;
    if (event.code !== 1000) {
      reconnectTimer = setTimeout(() => {
        console.log('[WS] Reconnecting...');
        connectPipeline(messageHandler, errorHandler);
      }, 3000);
    }
  };

  return socket;
}

export function sendPipelineMessage(payload) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    throw new Error('WebSocket not connected');
  }
  socket.send(JSON.stringify(payload));
}

export function sendTextForTranslation(text) {
  sendPipelineMessage({
    input_mode: 'text',
    text: text,
  });
}

export function sendAudioForTranslation(audioBase64) {
  sendPipelineMessage({
    input_mode: 'speech',
    audio_base64: audioBase64,
  });
}

export function disconnectPipeline() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (socket) {
    socket.close(1000);
    socket = null;
  }
}

export function isConnected() {
  return socket && socket.readyState === WebSocket.OPEN;
}
