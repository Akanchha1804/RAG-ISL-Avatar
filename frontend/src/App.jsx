import { useState, useCallback } from 'react';
import TextInput from './components/TextInput';
import MicRecorder from './components/MicRecorder';
import GlossDisplay from './components/GlossDisplay';
import AvatarView from './components/AvatarView';
import { fullPipeline } from './api';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('text');

  const handleTextResult = useCallback((data, isLoading, err) => {
    setLoading(isLoading);
    setError(err);
    if (data) setResult(data);
  }, []);

  const handleRecording = useCallback(async (blob) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fullPipeline(blob);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">
            ISL Avatar Translator
          </h1>
          <p className="text-blue-300">
            RAG-Augmented Text-to-Indian Sign Language
          </p>
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
                <TextInput onResult={handleTextResult} loading={loading} />
              ) : (
                <MicRecorder onTranscript={() => {}} onRecording={handleRecording} />
              )}

              {loading && (
                <div className="mt-4 flex items-center gap-2 text-blue-300">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <span>Processing...</span>
                </div>
              )}

              {error && (
                <div className="mt-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-sm">
                  {error}
                </div>
              )}
            </div>

            <GlossDisplay result={result} />
          </div>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <h2 className="text-lg font-semibold mb-4 text-center">Avatar Viewport</h2>
            <AvatarView landmarkUrl={result?.landmark_url || null} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
