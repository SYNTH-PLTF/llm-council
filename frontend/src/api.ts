import type { CouncilRun } from './types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')

// In dev the Vite proxy forwards /v1 to localhost:8000 (same origin). A deployed
// build has no proxy, so a backend URL must be configured via VITE_API_BASE_URL.
export function backendConfigured(): boolean {
  return import.meta.env.DEV || API_BASE.length > 0
}

export async function postChat(
  query: string,
  forceCouncil: boolean,
  token?: string,
): Promise<CouncilRun> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${API_BASE}/v1/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ query, force_council: forceCouncil }),
  })
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }
  return (await res.json()) as CouncilRun
}
