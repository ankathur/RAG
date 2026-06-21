import { useRef, useState } from 'react'

export default function IngestBar({ ingestFile, ingestPaths }) {
  const fileRef = useRef(null)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState(null) // { ok: bool, text: string }
  const [pathInput, setPathInput] = useState('')
  const [dragOver, setDragOver] = useState(false)

  async function run(fn) {
    setBusy(true)
    setStatus(null)
    try {
      const res = await fn()
      const ids = (res && res.ingested) || []
      setStatus({
        ok: true,
        text: ids.length
          ? `Ingested ${ids.length} document${ids.length > 1 ? 's' : ''}: ${ids.join(', ')}`
          : 'Ingest finished (no documents found).',
      })
    } catch (e) {
      setStatus({ ok: false, text: e.message || String(e) })
    } finally {
      setBusy(false)
    }
  }

  function onFiles(files) {
    if (!files || !files.length) return
    // The backend accepts one file per request — send them sequentially.
    run(async () => {
      let last
      for (const f of files) last = await ingestFile(f)
      return last
    })
  }

  function submitPaths() {
    const paths = pathInput
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
    if (!paths.length) return
    run(() => ingestPaths(paths)).then(() => setPathInput(''))
  }

  return (
    <section className="ingest">
      <div
        className={`dropzone ${dragOver ? 'drag' : ''} ${busy ? 'busy' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          onFiles(e.dataTransfer.files)
        }}
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          hidden
          multiple
          accept=".pdf,.md,.markdown,.txt,.text"
          onChange={(e) => onFiles(e.target.files)}
        />
        <span className="dz-main">
          <strong>Drop files</strong> or click to upload
        </span>
        <span className="hint">PDF · Markdown · text</span>
      </div>

      <div className="bypath">
        <input
          type="text"
          placeholder="…or ingest by server path (comma-separated)"
          value={pathInput}
          onChange={(e) => setPathInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') submitPaths()
          }}
        />
        <button disabled={busy || !pathInput.trim()} onClick={submitPaths}>
          Ingest path
        </button>
      </div>

      {busy && <p className="status muted">Indexing… building vector + pageindex</p>}
      {!busy && status && (
        <p className={`status ${status.ok ? 'ok' : 'err'}`}>{status.text}</p>
      )}
    </section>
  )
}
