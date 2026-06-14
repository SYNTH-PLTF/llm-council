import type { CouncilRun } from './types'

export async function postChat(
  query: string,
  forceCouncil: boolean,
  apiKey?: string,
): Promise<CouncilRun> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`
  const res = await fetch('/v1/chat', {
    method: 'POST',
    headers,
    body: JSON.stringify({ query, force_council: forceCouncil }),
  })
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }
  return (await res.json()) as CouncilRun
}
