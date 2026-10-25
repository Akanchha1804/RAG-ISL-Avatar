import { signedCoverage } from '../avatar/playback';

export default function GlossDisplay({ result, sourceInfo }) {
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

  // Composed playback signs one clip per gloss. A gloss whose entry has no
  // landmark clip is not signed at all (the avatar filters it out), so say so
  // instead of leaving the viewer to notice a missing word. Sentence-level
  // playback covers every token with one clip, so this counter only applies
  // when the playlist drives the avatar.
  const coverage = signedCoverage(
    glosses, result.animation, result.landmark_url);
  const { composed, signed: signedCount, missing } = coverage;
  // Sentence-level playback is one recorded clip; a playlist is the
  // experimental gloss-by-gloss path. Keep the distinction visible: the demo
  // must not imply that isolated gloss clips add up to fluent ISL.
  const mode = result.landmark_url ? 'sentence' : composed ? 'composed' : null;

  return (
    <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-3">
      <h3 className="font-semibold text-white">Translation Result</h3>

      {mode === 'sentence' && (
        <div className="rounded border border-emerald-400/30 bg-emerald-400/10 p-2 text-xs text-emerald-200">
          <span className="font-semibold">Recorded sentence clip.</span>{' '}
          Replaying one recorded ISL sentence.
        </div>
      )}
      {mode === 'composed' && (
        <div className="rounded border border-amber-400/30 bg-amber-400/10 p-2 text-xs text-amber-200">
          <span className="font-semibold">Experimental gloss playback.</span>{' '}
          Separate recorded gloss clips, not a fluent ISL translation.
        </div>
      )}
      {sourceInfo?.oneHanded && (
        <p className="text-xs text-amber-300">
          {sourceInfo.mode === 'composed'
            ? `${sourceInfo.oneHandedClips} of ${sourceInfo.clips} source clips are largely one-handed; the other hand rests or holds.`
            : 'One hand does most of the motion in this source clip; the other rests or holds.'}
        </p>
      )}

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

      {result.translation
        && (result.translation.translated_text || result.translation.warning) && (
        <div>
          {result.translation.translated_text && (
            <>
              <span className="text-xs text-slate-400 uppercase">
                Translated ({result.translation.detected_language} → en)
              </span>
              <p className="text-slate-100">{result.translation.translated_text}</p>
            </>
          )}
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
        {composed && glosses.length > 0 && (
          <span
            className={signedCount >= glosses.length ? 'text-emerald-300' : 'text-amber-300'}
          >
            {signedCount >= glosses.length
              ? `Playing all ${glosses.length} gloss clips`
              : `Playing ${signedCount} of ${glosses.length} gloss clips`}
          </span>
        )}
        {Array.isArray(result.animation?.unresolved_tokens) &&
          result.animation.unresolved_tokens.length > 0 && (
            <span className="text-amber-300">
              Not in vocabulary: {result.animation.unresolved_tokens.join(', ')}
            </span>
          )}
        {missing.length > 0 && (
          <span className="text-amber-300">
            Not signed (no motion yet): {missing.join(', ')}
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

      {(result.method || result.landmark_file || result.animation?.retrieval_detail) && (
        <details className="text-xs text-slate-500">
          <summary className="cursor-pointer text-slate-400">Technical details</summary>
          <div className="mt-1 flex flex-wrap gap-4">
            {result.method && <span>Method: {result.method}</span>}
            {result.landmark_file && <span>Landmarks: {result.landmark_file}</span>}
            {result.animation?.retrieval_detail && (
              <span>
                Retrieval: {result.animation.retrieval_detail.level}
                ({result.animation.retrieval_detail.store})
              </span>
            )}
          </div>
        </details>
      )}
    </div>
  );
}
