import { useState } from 'react';

export default function TextInput({ onTranslate, loading }) {
  const [text, setText] = useState('');
  const [useRag, setUseRag] = useState(true);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim() || loading) return;
    // App owns loading/error/result state for both the WS and REST paths.
    await onTranslate(text.trim(), useRag);
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type an English sentence to translate to ISL..."
        className="w-full h-24 p-3 border border-slate-600 rounded-lg resize-none bg-slate-900/60 text-white placeholder:text-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        disabled={loading}
      />
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
    </form>
  );
}
