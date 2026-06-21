export default function Message({ msg }) {
  const { role, text, citations, mode, error, usage } = msg
  const tokens = usage && (usage.total_tokens || usage.totalTokens)

  return (
    <div className={`msg ${role}`}>
      <div className={`bubble ${error ? 'error' : ''}`}>
        <div className="meta">
          {error && <span className="badge err">error</span>}
          {role === 'assistant' && !error && mode && (
            <span className="badge">{mode}</span>
          )}
        </div>

        <div className="text">{text}</div>

        {citations && citations.length > 0 && (
          <div className="citations">
            <div className="cit-title">Sources</div>
            <ol>
              {citations.map((c, i) => (
                <li key={i}>
                  <span className="cit-loc">{c.locator || c.doc_id}</span>
                  {c.origin && <span className={`tag ${c.origin}`}>{c.origin}</span>}
                  {typeof c.score === 'number' && (
                    <span className="cit-score">{c.score.toFixed(2)}</span>
                  )}
                  <span className="cit-doc muted">{c.doc_id}</span>
                </li>
              ))}
            </ol>
          </div>
        )}

        {tokens ? <div className="usage muted">{tokens} tokens</div> : null}
      </div>
    </div>
  )
}
