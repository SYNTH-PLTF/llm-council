import { Authenticator } from '@aws-amplify/ui-react'
import '@aws-amplify/ui-react/styles.css'
import { type ChangeEvent, type FormEvent, useState } from 'react'

import { backendConfigured, postChat } from './api'
import { CouncilResult } from './components/CouncilResult'
import type { CouncilRun } from './types'

function Council({ username, onSignOut }: { username?: string; onSignOut?: () => void }) {
  const [query, setQuery] = useState('')
  const [forceCouncil, setForceCouncil] = useState(false)
  const [run, setRun] = useState<CouncilRun | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const configured = backendConfigured()

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    if (!query.trim() || !configured) return
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
      <header className="topbar">
        <h1>AI Council</h1>
        <div className="who">
          <span>{username}</span>
          <button type="button" onClick={onSignOut}>
            Sign out
          </button>
        </div>
      </header>
      <p className="tagline">A selective multi-LLM council: routed, debiased, cost-controlled.</p>
      {!configured && (
        <div className="banner" role="alert">
          Backend not configured. Set <code>VITE_API_BASE_URL</code> to your council API and redeploy.
        </div>
      )}
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
          <button type="submit" disabled={loading || !configured}>
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </div>
      </form>
      {error && <div className="error">request failed: {error}</div>}
      {run && <CouncilResult run={run} />}
    </main>
  )
}

export function App() {
  return (
    <Authenticator loginMechanisms={['username']} hideSignUp>
      {({ signOut, user }) => <Council username={user?.username} onSignOut={signOut} />}
    </Authenticator>
  )
}
