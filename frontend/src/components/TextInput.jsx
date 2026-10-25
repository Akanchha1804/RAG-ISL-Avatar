import { useState } from 'react';

// One-click demo sentences. Every entry is a curated corpus sentence whose
// glosses all have landmark clips (verified against the running backend), so
// a click cannot land on "Signed 0 of N" in front of a judge. The first four
// are the hackathon demo set, ordered for a short run; the rest are extras.
// The Hindi entry also demonstrates the hi->en translation stage.
const EXAMPLES = [
  'hi how are you',
  'help me',
  'do not worry',
  'congratulations',
  'are you free today',
  'can you repeat that please',
  'सुप्रभात',
];

export default function TextInput({ onTranslate, loading }) {
  const [text, setText] = useState('');
  const [useRag, setUseRag] = useState(true);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim() || loading) return;
    // App owns loading/error/result state for both the WS and REST paths.
    await onTranslate(text.trim(), useRag);
  };

  const tryExample = (sentence) => {
    if (loading) return;
    setText(sentence);
    onTranslate(sentence, useRag);
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type in English, or Hindi in Devanagari (e.g. सुप्रभात)..."
        className="w-full h-24 p-3 border border-slate-600 rounded-lg resize-none bg-slate-900/60 text-white placeholder:text-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        disabled={loading}
      />
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs uppercase tracking-wide text-slate-400">Try</span>
        {EXAMPLES.map((sentence) => (
          <button
            key={sentence}
            type="button"
            onClick={() => tryExample(sentence)}
            disabled={loading}
            className="px-2.5 py-1 rounded-full border border-slate-600 text-xs text-slate-200 hover:border-blue-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {sentence}
          </button>
        ))}
      </div>
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={useRag}
            onChange={(e) => setUseRag(e.target.checked)}
            className="rounded"
          />
          Use RAG retrieval
        </label>
        <button
          type="submit"
          disabled={!text.trim() || loading}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-slate-600 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Translating...' : 'Translate to ISL'}
        </button>
      </div>
      <p className="text-xs text-slate-400">
        Hindi is translated to English before glossing. The first translation
        can take ~30 s while the models warm up; after that it is a few seconds.
      </p>
    </form>
  );
}
