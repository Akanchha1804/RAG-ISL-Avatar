// Single source of truth for where the backend lives.
//
// Default is same-origin, which works in both environments because every
// request is proxied:
//   - dev:   vite dev server proxies /api, /landmarks, /clips and
//            /avatar (see vite.config.js)
//   - prod:  nginx proxies them (see nginx.conf)
// Set VITE_API_URL to point at a backend on another host.
//
// REST and WS previously disagreed (REST defaulted to same-origin while WS
// hardcoded http://localhost:8000), so a deployed frontend could fetch JSON
// through the proxy but never open its WebSocket.

export const API_BASE = import.meta.env.VITE_API_URL || '';

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

export function wsUrl(path) {
  if (API_BASE) {
    return API_BASE.replace(/^http/, 'ws') + path;
  }
  const loc = window.location;
  const scheme = loc.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${loc.host}${path}`;
}
