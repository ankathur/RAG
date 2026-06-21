const MODES = [
  { value: '', label: 'Default' },
  { value: 'vector', label: 'Vector' },
  { value: 'pageindex', label: 'PageIndex' },
  { value: 'hybrid', label: 'Hybrid' },
]

export default function Header({
  health,
  healthError,
  onRefresh,
  mode,
  setMode,
  topK,
  setTopK,
  onOpenSettings,
}) {
  const status = healthError
    ? { cls: 'down', text: 'API unreachable' }
    : health?.llm_reachable
      ? { cls: 'ok', text: 'LLM ready' }
      : health
        ? { cls: 'warn', text: 'LLM unreachable' }
        : { cls: 'warn', text: 'checking…' }

  return (
    <header className="header">
      <div className="brand">
        <span className="logo">◆</span>
        <div>
          <h1>RAG System</h1>
          <p className="subtitle">vector · pageindex · hybrid retrieval</p>
        </div>
      </div>

      <div className="controls">
        <label className="field">
          <span>Mode</span>
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            {MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>top_k</span>
          <input
            type="number"
            min="1"
            max="50"
            placeholder="auto"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
          />
        </label>

        <button className="health" onClick={onRefresh} title="Refresh status">
          <span className={`dot ${status.cls}`} />
          {status.text}
          {health?.mode && <span className="muted"> · {health.mode}</span>}
        </button>

        <button className="gear" onClick={onOpenSettings} title="LLM settings">
          ⚙
        </button>
      </div>
    </header>
  )
}
