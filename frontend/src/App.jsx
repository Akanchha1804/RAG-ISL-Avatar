import { useState, useCallback, useEffect, useRef } from 'react';
import TextInput from './components/TextInput';
import MicRecorder from './components/MicRecorder';
import GlossDisplay from './components/GlossDisplay';
import AvatarView from './components/AvatarView';
import { fullPipeline, translateText } from './api';
import {
  connectPipeline,
  sendTextForTranslation,
  sendAudioForTranslation,
  disconnectPipeline,
  isConnected,
} from './ws';

// A pipeline run that never reports back must not spin forever.
// 180s: first voice request pays the STT cold-start (model + VAD load on
// CPU); warm requests finish in seconds. Do not lower this without first
// warming the STT model at backend startup.
const REQUEST_TIMEOUT_MS = 180000;

const STAGE_ORDER = ['transcribing', 'retrieving', 'generating', 'animating'];

const stageLabel = {
  transcribing: 'Transcribing speech...',
  retrieving: 'Searching for ISL match...',
  generating: 'Generating ISL glosses...',
  animating: 'Loading avatar animation...',
};

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('text');
  const [pipelineStage, setPipelineStage] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const timeoutRef = useRef(null);
  const handleWsMessageRef = useRef(null);

  const clearPending = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const fail = useCallback(
    (message) => {
      clearPending();
      setLoading(false);
      setPipelineStage(null);
      setError(message);
    },
    [clearPending],
  );

  const handleWsMessage = useCallback(
    (data) => {
      // Errors arrive as {stage, status:'error', message} and the terminal
      // frame as {stage:'complete', status:'error', result:{error}}. Without
      // this the UI rendered an empty card showing "Method: undefined".
      if (data.status === 'error') {
        const message = data.message || data.result?.error;
        if (data.stage === 'complete' || message) {
          fail(message || 'Pipeline failed');
        }
        return;
      }

      if (data.stage === 'complete') {
        clearPending();
        setLoading(false);
        setPipelineStage(null);
        setError(null);
        setResult(data.result);
        return;
      }

      if (STAGE_ORDER.includes(data.stage)) {
        setPipelineStage(data.stage);
        if (data.status === 'done') {
          const next = STAGE_ORDER[STAGE_ORDER.indexOf(data.stage) + 1];
          setPipelineStage(next || null);
        }
      }
    },
    [clearPending, fail],
  );

  // Keep the latest handler reachable from the connection effect without
  // reconnecting every time it changes.
  useEffect(() => {
    handleWsMessageRef.current = handleWsMessage;
  }, [handleWsMessage]);

  useEffect(() => {
    connectPipeline({
      onMessage: (data) => handleWsMessageRef.current(data),
      onError: () => setWsConnected(false),
      onOpen: () => setWsConnected(true),
      onClose: () => setWsConnected(false),
    });

    return () => disconnectPipeline();
  }, []);

  // Release the timeout if the component unmounts mid-request.
  useEffect(() => clearPending, [clearPending]);

  const startRequest = useCallback(() => {
    clearPending();
    setLoading(true);
    setError(null);
    setResult(null);
    setPipelineStage('transcribing');
    timeoutRef.current = setTimeout(() => {
      fail('The request timed out. Is the backend running on port 8000?');
    }, REQUEST_TIMEOUT_MS);
  }, [clearPending, fail]);

  // Text input prefers the WebSocket (it streams stage progress) and falls
  // back to REST when the socket is unavailable.
  const handleTextTranslate = useCallback(
    async (text, useRag) => {
      startRequest();
      if (isConnected()) {
        try {
          sendTextForTranslation(text, useRag);
          return;
        } catch (err) {
          console.warn('[App] WS send failed, falling back to REST:', err);
        }
      }
      try {
        const data = await translateText(text, useRag);
        clearPending();
        setLoading(false);
        setPipelineStage(null);
        setResult(data);
      } catch (err) {
        fail(err.message);
      }
    },
    [startRequest, clearPending, fail],
  );

  const handleRecording = useCallback(
    async (audioBase64) => {
      startRequest();
      if (isConnected()) {
        try {
          sendAudioForTranslation(audioBase64);
          return;
        } catch (err) {
          console.warn('[App] WS send failed, falling back to REST:', err);
        }
      }

      try {
        const bytes = Uint8Array.from(atob(audioBase64), (c) => c.charCodeAt(0));
        const file = new File([bytes], 'audio.webm', { type: 'audio/webm' });
        const data = await fullPipeline(file);
        clearPending();
        setLoading(false);
        setPipelineStage(null);
        setResult(data);
      } catch (err) {
        fail(err.message);
      }
    },
    [startRequest, clearPending, fail],
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">ISL Avatar Translator</h1>
          <p className="text-blue-300">RAG-Augmented Text-to-Indian Sign Language</p>
          <div className="mt-2 flex items-center justify-center gap-2 text-sm">
            <span
              className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-yellow-400'}`}
              aria-hidden="true"
            />
            <span className="text-gray-400">
              {wsConnected ? 'Real-time connected' : 'REST mode'}
            </span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur rounded-xl p-6">
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => setMode('text')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    mode === 'text'
                      ? 'bg-blue-500 text-white'
                      : 'bg-white/10 text-gray-300 hover:bg-white/20'
                  }`}
                >
                  Text Input
                </button>
                <button
                  onClick={() => setMode('voice')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    mode === 'voice'
                      ? 'bg-blue-500 text-white'
                      : 'bg-white/10 text-gray-300 hover:bg-white/20'
                  }`}
                >
                  Voice Input
                </button>
              </div>

              {mode === 'text' ? (
                <TextInput onTranslate={handleTextTranslate} loading={loading} />
              ) : (
                <MicRecorder onAudioReady={handleRecording} />
              )}

              {loading && (
                <div className="mt-4 space-y-2">
                  <div className="flex items-center gap-2 text-blue-300">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" aria-hidden="true">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                      />
                    </svg>
                    <span>
                      {pipelineStage ? stageLabel[pipelineStage] : 'Processing...'}
                    </span>
                  </div>
                  {pipelineStage && (
                    <div className="flex gap-1">
                      {STAGE_ORDER.map((stage, i) => (
                        <div
                          key={stage}
                          className={`h-1 flex-1 rounded-full transition-colors duration-300 ${
                            STAGE_ORDER.indexOf(pipelineStage) >= i
                              ? 'bg-blue-400'
                              : 'bg-white/10'
                          }`}
                        />
                      ))}
                    </div>
                  )}
                </div>
              )}

              {error && (
                <div
                  role="alert"
                  className="mt-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-sm"
                >
                  {error}
                </div>
              )}
            </div>

            <GlossDisplay result={result} />
          </div>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <h2 className="text-lg font-semibold mb-4 text-center">Avatar Viewport</h2>
            <AvatarView landmarkUrl={result?.landmark_url || null} animation={result?.animation || null} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
