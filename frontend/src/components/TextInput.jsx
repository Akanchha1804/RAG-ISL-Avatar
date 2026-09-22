import { useState } from 'react';
import { translateText } from '../api';

export default function TextInput({ onResult, loading }) {
  const [text, setText] = useState('');
  const [useRag, setUseRag] = useState(true);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim() || loading) return;
    onResult(null, true);
    try {
      const result = await translateText(text.trim(), useRag);
      onResult(result, false);
    } catch (err) {
      onResult(null, false, err.message);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type an English sentence to translate to ISL..."
        className="w-full h-24 p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        disabled={loading}
      />
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm text-gray-600">
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
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Translating...' : 'Translate to ISL'}
        </button>
      </div>
    </form>
  );
}
