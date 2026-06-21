import { useEffect, useState } from 'react'
import {
  getHealth,
  ask,
  ingestFile,
  ingestPaths,
  getConfig,
  updateConfig,
  ApiError,
} from './api.js'
import Header from './components/Header.jsx'
import IngestBar from './components/IngestBar.jsx'
import Chat from './components/Chat.jsx'
import Settings from './components/Settings.jsx'

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(false)
  const [mode, setMode] = useState('') // '' = use the server's default mode
  const [topK, setTopK] = useState('') // '' = use the server's default top_k
  const [messages, setMessages] = useState([])
  const [asking, setAsking] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [notice, setNotice] = useState(null)

  async function refreshHealth() {
    try {
      setHealth(await getHealth())
      setHealthError(false)
    } catch {
      setHealth(null)
      setHealthError(true)
    }
  }

  useEffect(() => {
    refreshHealth()
  }, [])

  async function handleAsk(query) {
    setMessages((m) => [...m, { role: 'user', text: query }])
    setAsking(true)
    try {
      const res = await ask({
        query,
        mode: mode || undefined,
        topK: topK ? Number(topK) : undefined,
      })
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text: res.answer,
          citations: res.citations || [],
          mode: res.mode,
          usage: res.usage,
        },
      ])
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String(e)
      setMessages((m) => [...m, { role: 'assistant', error: true, text: msg }])
    } finally {
      setAsking(false)
    }
  }

  function handleSaved(res) {
    // Reflect the freshly-applied config in the health badge…
    if (res && typeof res.llm_reachable === 'boolean') {
      setHealth((h) => ({ ...(h || { status: 'ok' }), llm_reachable: res.llm_reachable }))
    }
    refreshHealth()
    setNotice(
      res?.embedding_changed
        ? {
            kind: 'warn',
            text: 'Embeddings changed — re-ingest your documents so the vector index matches.',
          }
        : {
            kind: 'ok',
            text: res?.persisted
              ? 'Settings applied and saved to .env.'
              : 'Settings applied for this session.',
          },
    )
  }

  return (
    <div className="app">
      <Header
        health={health}
        healthError={healthError}
        onRefresh={refreshHealth}
        mode={mode}
        setMode={setMode}
        topK={topK}
        setTopK={setTopK}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      {notice && (
        <div className={`banner ${notice.kind}`}>
          <span>{notice.text}</span>
          <button className="icon" onClick={() => setNotice(null)}>
            ✕
          </button>
        </div>
      )}

      <IngestBar ingestFile={ingestFile} ingestPaths={ingestPaths} />
      <Chat messages={messages} asking={asking} onAsk={handleAsk} />

      {settingsOpen && (
        <Settings
          getConfig={getConfig}
          updateConfig={updateConfig}
          onClose={() => setSettingsOpen(false)}
          onSaved={handleSaved}
        />
      )}
    </div>
  )
}
