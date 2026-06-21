// Thin client for the RAG System REST API (see SPEC.md §12).
// Base URL is empty in dev (Vite proxy); set VITE_API_BASE_URL for prod builds.
const BASE = import.meta.env.VITE_API_BASE_URL || ''

export class ApiError extends Error {
  constructor(message, type, status) {
    super(message)
    this.name = 'ApiError'
    this.type = type
    this.status = status
  }
}

// Parse a response, raising ApiError with the backend's {error:{type,message}}.
async function parse(res) {
  let body = null
  try {
    body = await res.json()
  } catch {
    // non-JSON (e.g. proxy/network failure) — handled below
  }
  if (!res.ok) {
    const err = body && body.error
    throw new ApiError(
      (err && err.message) || res.statusText || 'Request failed',
      (err && err.type) || 'http_error',
      res.status,
    )
  }
  return body
}

async function send(path, options) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, options)
  } catch (e) {
    throw new ApiError(
      `Could not reach the API${BASE ? ` at ${BASE}` : ''}. Is the backend running?`,
      'network_error',
      0,
    )
  }
  return parse(res)
}

export function getHealth() {
  return send('/health', {})
}

export function ingestFile(file) {
  const form = new FormData()
  form.append('file', file)
  return send('/ingest', { method: 'POST', body: form })
}

export function ingestPaths(paths) {
  return send('/ingest', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ paths }),
  })
}

export function getConfig() {
  return send('/config', {})
}

export function updateConfig(payload) {
  return send('/config', {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function ask({ query, mode, topK }) {
  const payload = { query }
  if (mode) payload.mode = mode
  if (topK) payload.top_k = topK
  return send('/ask', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
}
