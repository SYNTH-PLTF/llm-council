import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { CouncilRun } from '../types'
import { CouncilResult } from './CouncilResult'

const councilRun: CouncilRun = {
  correlation_id: 'abc',
  final_answer: 'The synthesized answer.',
  decision: 'council',
  query_class: 'high_stakes',
  confidence: 'high',
  dissent_notes: 'P2 disagreed on the cost estimate.',
  contributing_sources: ['1', '2'],
  disagreement: 0.42,
  degraded: false,
  timeout_partial: false,
  cost_usd: 0.0123,
  latency_ms: 4200,
  flags: [],
  stages: ['triage', 'proposers', 'ranking', 'chairman'],
  proposer_models: ['claude-opus-4-8', 'gpt-5.4', 'grok-4.3'],
}

const singleRun: CouncilRun = {
  ...councilRun,
  decision: 'single_model',
  query_class: 'trivial',
  dissent_notes: '',
  disagreement: 0,
  stages: ['triage', 'single_model'],
  proposer_models: ['claude-opus-4-8'],
}

describe('CouncilResult', () => {
  it('renders a full council run with badges, proposers, disagreement, dissent', () => {
    render(<CouncilResult run={councilRun} />)
    expect(screen.getByText('council')).toBeInTheDocument()
    expect(screen.getByText('high_stakes')).toBeInTheDocument()
    expect(screen.getByText('disagreement')).toBeInTheDocument()
    expect(screen.getByText(/P2 disagreed/)).toBeInTheDocument()
    for (const model of councilRun.proposer_models) {
      expect(screen.getByText(model)).toBeInTheDocument()
    }
  })

  it('renders a single-model run without empty council sections', () => {
    render(<CouncilResult run={singleRun} />)
    expect(screen.getAllByText('single_model').length).toBeGreaterThan(0)
    expect(screen.queryByText('disagreement')).not.toBeInTheDocument()
    expect(screen.queryByText('Proposers')).not.toBeInTheDocument()
  })
})
