import { useEffect, useRef, useState } from 'react'
import Message from './Message.jsx'

export default function Chat({ messages, asking, onAsk }) {
  const [text, setText] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, asking])

  function submit() {
    const q = text.trim()
    if (!q || asking) return
    onAsk(q)
    setText('')
  }

  return (
    <section className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ingest a document, then ask a question.</p>
            <p className="muted">
              Answers are grounded in your sources and cite where they came from.
            </p>
          </div>
        )}

        {messages.map((m, i) => (
          <Message key={i} msg={m} />
        ))}

        {asking && (
          <div className="msg assistant">
            <div className="bubble typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}

        <div ref={endRef} />
      </div>

      <div className="composer">
        <textarea
          rows={1}
          placeholder="Ask a question…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
        />
        <button disabled={!text.trim() || asking} onClick={submit}>
          {asking ? '…' : 'Ask'}
        </button>
      </div>
    </section>
  )
}
