export default function GlossDisplay({ result }) {
  if (!result) return null;

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-3">
      <h3 className="font-semibold text-gray-700">Translation Result</h3>

      {result.input_text && (
        <div>
          <span className="text-xs text-gray-500 uppercase">Input</span>
          <p className="text-gray-800">{result.input_text}</p>
        </div>
      )}

      {result.transcript && (
        <div>
          <span className="text-xs text-gray-500 uppercase">Transcript</span>
          <p className="text-gray-800">{result.transcript}</p>
        </div>
      )}

      {result.matched_sentence && (
        <div>
          <span className="text-xs text-gray-500 uppercase">Matched Sentence</span>
          <p className="text-gray-800 font-medium">{result.matched_sentence}</p>
          {result.similarity && (
            <span className="text-xs text-blue-500">
              Similarity: {(result.similarity * 100).toFixed(1)}%
            </span>
          )}
        </div>
      )}

      {result.glosses && (
        <div>
          <span className="text-xs text-gray-500 uppercase">ISL Glosses</span>
          <div className="flex flex-wrap gap-2 mt-1">
            {result.glosses.split(' ').map((gloss, i) => (
              <span
                key={i}
                className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm font-mono"
              >
                {gloss}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="flex items-center gap-4 text-xs text-gray-400">
        <span>Method: {result.method}</span>
        {result.landmark_file && (
          <span>Landmarks: {result.landmark_file}</span>
        )}
      </div>
    </div>
  );
}
