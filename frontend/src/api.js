const API_BASE = import.meta.env.VITE_API_URL || '';

async function fetchJSON(url, options = {}) {
  const res = await fetch(`${API_BASE}${url}`, options);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API Error ${res.status}: ${err}`);
  }
  return res.json();
}

export async function translateText(text, useRag = true) {
  return fetchJSON('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, use_rag: useRag }),
  });
}

export async function transcribeAudio(blob) {
  const formData = new FormData();
  formData.append('file', blob, 'audio.wav');
  return fetchJSON('/api/transcribe', {
    method: 'POST',
    body: formData,
  });
}

export async function fullPipeline(blob) {
  const formData = new FormData();
  formData.append('file', blob, 'audio.wav');
  return fetchJSON('/api/pipeline', {
    method: 'POST',
    body: formData,
  });
}

export async function getSentences() {
  return fetchJSON('/api/sentences');
}

export async function searchSentences(query) {
  return fetchJSON(`/api/search?q=${encodeURIComponent(query)}&top_k=5`);
}

export async function getGlossVocabulary(limit = 100) {
  return fetchJSON(`/api/gloss-vocabulary?limit=${limit}`);
}

export async function healthCheck() {
  return fetchJSON('/api/health');
}
