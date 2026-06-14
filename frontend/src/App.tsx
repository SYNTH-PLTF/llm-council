import { type ChangeEvent, type FormEvent, useState } from 'react'

import { postChat } from './api'
import { CouncilResult } from './components/CouncilResult'
import type { CouncilRun } from './types'

export function App() {
  const [query, setQuery] = useState('')
  const [forceCouncil, setForceCouncil] = useState(false)
  const [run, setRun] = useState<CouncilRun | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setRun(null)
    try {
      setRun(await postChat(query, forceCouncil))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <h1>AI Council</h1>
      <p className="tagline">A selective multi-LLM council: routed, debiased, cost-controlled.</p>
      <form onSubmit={onSubmit}>
        <textarea
          value={query}
          onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setQuery(e.target.value)}
          placeholder="Ask a hard question..."
          rows={3}
        />
        <div className="controls">
          <label>
            <input
              type="checkbox"
              checked={forceCouncil}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setForceCouncil(e.target.checked)}
            />{' '}
            force council
          </label>
          <button type="submit" disabled={loading}>
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </div>
      </form>
      {error && <div className="error">request failed: {error}</div>}
      {run && <CouncilResult run={run} />}
    </main>
  )
}
