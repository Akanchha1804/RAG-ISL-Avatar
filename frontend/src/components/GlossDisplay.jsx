export default function GlossDisplay({ result }) {
  if (!result) return null;

  // Glosses arrive as a string from the API; tolerate an array too so a
  // schema change cannot crash the panel. Prefer the token sequence when
  // present so a multi-word gloss (e.g. "THANK YOU") renders as one sign.
  const glosses = Array.isArray(result.gloss_sequence) && result.gloss_sequence.length > 0
    ? result.gloss_sequence.map((g) => String(g))
    : Array.isArray(result.glosses)
      ? result.glosses
      : String(result.glosses || '')
          .split(/\s+/)
          .filter(Boolean);

  const similarity =
    typeof result.similarity === 'number' ? result.similarity : null;

  return (
    <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-3">
      <h3 className="font-semibold text-white">Translation Result</h3>

      {result.input_text && (
        <div>
          <span className="text-xs text-slate-400 uppercase">Input</span>
          <p className="text-slate-100">{result.input_text}</p>
        </div>
      )}

      {result.transcript && result.transcript !== result.input_text && (
        <div>
          <span className="text-xs text-slate-400 uppercase">Transcript</span>
          <p className="text-slate-100">{result.transcript}</p>
          {result.stt && (result.stt.language || result.stt.duration) && (
            <span className="text-xs text-slate-400">
              {result.stt.language ? `Language: ${result.stt.language}` : ''}
              {result.stt.language_probability
                ? ` (${(result.stt.language_probability * 100).toFixed(0)}%)`
                : ''}
              {result.stt.duration ? ` · ${result.stt.duration}s` : ''}
            </span>
          )}
          {result.stt && typeof result.stt.transcript_confidence === 'number'
            && result.stt.transcript_confidence < 0.5 && (
            <span className="text-xs text-amber-300">
              Low transcription confidence — please speak clearly and retry.
            </span>
          )}
        </div>
      )}

      {result.translation && result.translation.translated_text && (
        <div>
          <span className="text-xs text-slate-400 uppercase">
            Translated ({result.translation.detected_language} → en)
          </span>
          <p className="text-slate-100">{result.translation.translated_text}</p>
          {result.translation.warning && (
            <span className="text-xs text-amber-300">{result.translation.warning}</span>
          )}
        </div>
      )}
      {result.translation && result.translation.detected_language === 'unsupported' && (
        <div>
          <span className="text-xs text-amber-300">
            This language is not supported yet — English and Hindi (Devanagari) only.
          </span>
        </div>
      )}
      {result.translation && result.translation.detected_language === 'hi-latn' && (
        <div>
          <span className="text-xs text-amber-300">
            Romanized Hindi is not translated yet — please type in Devanagari or English.
          </span>
        </div>
      )}

      {result.matched_sentence && (
        <div>
          <span className="text-xs text-slate-400 uppercase">Matched Sentence</span>
          <p className="text-slate-100 font-medium">{result.matched_sentence}</p>
          {similarity !== null && (
            <span className="text-xs text-blue-400">
              Similarity: {(similarity * 100).toFixed(1)}%
            </span>
          )}
        </div>
      )}

      {glosses.length > 0 && (
        <div>
          <span className="text-xs text-slate-400 uppercase">ISL Glosses</span>
          <div className="flex flex-wrap gap-2 mt-1">
            {glosses.map((gloss, i) => (
              <span
                key={i}
                className="px-2 py-1 bg-blue-400/20 text-blue-200 rounded text-sm font-mono"
              >
                {gloss}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400">
        {result.method && <span>Method: {result.method}</span>}
        {result.landmark_file && <span>Landmarks: {result.landmark_file}</span>}
        {result.animation?.retrieval_detail && (
          <span>
            Retrieval: {result.animation.retrieval_detail.level}
            ({result.animation.retrieval_detail.store})
          </span>
        )}
        {Array.isArray(result.animation?.unresolved_tokens) &&
          result.animation.unresolved_tokens.length > 0 && (
            <span className="text-amber-300">
              Unresolved glosses: {result.animation.unresolved_tokens.length}
            </span>
          )}
        {Array.isArray(result.animation?.unsupported_tokens) &&
          result.animation.unsupported_tokens.length > 0 && (
            <span className="text-amber-300">
              No clip for: {result.animation.unsupported_tokens.join(', ')}
            </span>
          )}
        {result.animation?.suggestions &&
          Object.keys(result.animation.suggestions).length > 0 && (
            <span className="text-slate-400">
              Did you mean:{' '}
              {Object.entries(result.animation.suggestions)
                .map(([t, s]) => `${t} → ${s.join('/')}`)
                .join('; ')}
            </span>
          )}
      </div>
    </div>
  );
}
