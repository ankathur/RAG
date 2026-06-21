import { useEffect, useState } from 'react'

const STRUCTURED = ['auto', 'json_schema', 'json_object', 'prompt']

// Convenience presets that fill base_url (+ a placeholder key) for an LLM role.
const PRESETS = {
  'Local Ollama': { base_url: 'http://localhost:11434/v1', api_key: 'ollama' },
  OpenAI: { base_url: 'https://api.openai.com/v1', api_key: '' },
  OpenRouter: { base_url: 'https://openrouter.ai/api/v1', api_key: '' },
}

export default function Settings({ getConfig, updateConfig, onClose, onSaved }) {
  const [cfg, setCfg] = useState(null)
  const [form, setForm] = useState(null)
  const [persist, setPersist] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    getConfig()
      .then((c) => {
        if (!alive) return
        setCfg(c)
        setForm({
          generation: blankRole(c.generation),
          reasoning: blankRole(c.reasoning),
          embedding: {
            provider: c.embedding.provider || 'local',
            model: c.embedding.model || '',
            base_url: c.embedding.base_url || '',
            api_key: '',
          },
          structured_output_mode: c.structured_output_mode || 'auto',
        })
      })
      .catch((e) => setError(e.message || String(e)))
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  function upd(section, field, value) {
    setForm((f) => ({ ...f, [section]: { ...f[section], [field]: value } }))
  }

  function applyPreset(section, name) {
    const p = PRESETS[name]
    if (!p) return
    setForm((f) => ({
      ...f,
      [section]: { ...f[section], base_url: p.base_url, api_key: p.api_key },
    }))
  }

  async function save() {
    setSaving(true)
    setError(null)
    try {
      const res = await updateConfig({ ...form, persist })
      onSaved(res)
      onClose()
    } catch (e) {
      setError(e.message || String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>LLM &amp; embedding settings</h2>
          <button className="icon" onClick={onClose} title="Close">
            ✕
          </button>
        </div>

        {!form && !error && <p className="muted">Loading…</p>}
        {error && <p className="status err">{error}</p>}

        {form && (
          <div className="settings-body">
            <RoleFields
              title="Generation"
              hint="answer synthesis"
              data={form.generation}
              apiSet={cfg?.generation.api_key_set}
              onChange={(f, v) => upd('generation', f, v)}
              onPreset={(name) => applyPreset('generation', name)}
            />
            <RoleFields
              title="Reasoning"
              hint="tree build · search · rerank"
              data={form.reasoning}
              apiSet={cfg?.reasoning.api_key_set}
              onChange={(f, v) => upd('reasoning', f, v)}
              onPreset={(name) => applyPreset('reasoning', name)}
            />

            <fieldset className="role">
              <legend>
                Embeddings <span className="muted">vector search</span>
              </legend>
              <label className="srow">
                <span>provider</span>
                <select
                  value={form.embedding.provider}
                  onChange={(e) => upd('embedding', 'provider', e.target.value)}
                >
                  <option value="local">local (sentence-transformers)</option>
                  <option value="openai">openai-compatible</option>
                </select>
              </label>
              <label className="srow">
                <span>model</span>
                <input
                  value={form.embedding.model}
                  onChange={(e) => upd('embedding', 'model', e.target.value)}
                  placeholder="all-MiniLM-L6-v2"
                />
              </label>
              {form.embedding.provider === 'openai' && (
                <>
                  <label className="srow">
                    <span>base_url</span>
                    <input
                      value={form.embedding.base_url}
                      onChange={(e) => upd('embedding', 'base_url', e.target.value)}
                      placeholder="https://…/v1"
                    />
                  </label>
                  <label className="srow">
                    <span>api_key</span>
                    <input
                      type="password"
                      value={form.embedding.api_key}
                      onChange={(e) => upd('embedding', 'api_key', e.target.value)}
                      placeholder={cfg?.embedding.api_key_set ? '•••• unchanged' : 'api key'}
                    />
                  </label>
                </>
              )}
              <p className="warn-text">
                ⚠ Changing embeddings invalidates the existing vector index —
                re-ingest your documents afterward.
              </p>
            </fieldset>

            <label className="srow">
              <span>structured output</span>
              <select
                value={form.structured_output_mode}
                onChange={(e) =>
                  setForm((f) => ({ ...f, structured_output_mode: e.target.value }))
                }
              >
                {STRUCTURED.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>

            <label className="persist">
              <input
                type="checkbox"
                checked={persist}
                onChange={(e) => setPersist(e.target.checked)}
              />
              Save to <code>.env</code> (survives restart)
            </label>

            <div className="modal-actions">
              <button onClick={onClose}>Cancel</button>
              <button className="primary" disabled={saving} onClick={save}>
                {saving ? 'Saving…' : 'Save & apply'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function blankRole(c) {
  // api_key starts blank — submitting blank keeps the existing key.
  return { base_url: c.base_url || '', model: c.model || '', api_key: '' }
}

function RoleFields({ title, hint, data, apiSet, onChange, onPreset }) {
  return (
    <fieldset className="role">
      <legend>
        {title} <span className="muted">{hint}</span>
      </legend>
      <div className="presets">
        {Object.keys(PRESETS).map((p) => (
          <button key={p} type="button" className="chip" onClick={() => onPreset(p)}>
            {p}
          </button>
        ))}
      </div>
      <label className="srow">
        <span>base_url</span>
        <input
          value={data.base_url}
          onChange={(e) => onChange('base_url', e.target.value)}
          placeholder="http://localhost:11434/v1"
        />
      </label>
      <label className="srow">
        <span>model</span>
        <input
          value={data.model}
          onChange={(e) => onChange('model', e.target.value)}
          placeholder="qwen2.5:7b-instruct"
        />
      </label>
      <label className="srow">
        <span>api_key</span>
        <input
          type="password"
          value={data.api_key}
          onChange={(e) => onChange('api_key', e.target.value)}
          placeholder={apiSet ? '•••• unchanged' : 'api key'}
        />
      </label>
    </fieldset>
  )
}
