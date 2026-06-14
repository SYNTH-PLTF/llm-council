export interface CouncilRun {
  correlation_id: string
  final_answer: string
  decision: string
  query_class: string
  confidence: string
  dissent_notes: string
  contributing_sources: string[]
  disagreement: number
  degraded: boolean
  timeout_partial: boolean
  cost_usd: number
  latency_ms: number
  flags: string[]
  stages: string[]
  proposer_models: string[]
}

export function isCouncil(run: CouncilRun): boolean {
  return run.decision === 'council' || run.decision === 'council_with_voting'
}
