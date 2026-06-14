import { isCouncil, type CouncilRun } from '../types'
import { Typewriter } from './Typewriter'

function Badge({ label, value, tone = 'default' }: { label: string; value: string; tone?: string }) {
  return (
    <span className={`badge badge-${tone}`}>
      <span className="badge-label">{label}</span>
      <span className="badge-value">{value}</span>
    </span>
  )
}

export function CouncilResult({ run }: { run: CouncilRun }) {
  const council = isCouncil(run)
  return (
    <div className="result">
      <div className="badges">
        <Badge label="route" value={run.decision} tone={council ? 'council' : 'single'} />
        <Badge label="class" value={run.query_class} />
        <Badge label="confidence" value={run.confidence} />
        <Badge label="cost" value={`$${run.cost_usd.toFixed(4)}`} />
        <Badge label="latency" value={`${Math.round(run.latency_ms)} ms`} />
        {run.degraded && <Badge label="degraded" value="yes" tone="warn" />}
        {run.timeout_partial && <Badge label="partial" value="timeout" tone="warn" />}
      </div>

      {run.flags.length > 0 && (
        <div className="flags" role="alert">
          guardrail flags: {run.flags.join(', ')}
        </div>
      )}

      <div className="stages">
        {run.stages.map((stage) => (
          <span key={stage} className="stage-pill">
            {stage}
          </span>
        ))}
      </div>

      {council && (
        <section className="proposers">
          <h4>Proposers</h4>
          <div className="proposer-tabs">
            {run.proposer_models.map((model) => (
              <span key={model} className="proposer-chip">
                {model}
              </span>
            ))}
          </div>
        </section>
      )}

      {council && (
        <section className="disagreement">
          <span className="disagreement-label">disagreement</span>
          <div className="bar">
            <div className="bar-fill" style={{ width: `${Math.round(run.disagreement * 100)}%` }} />
          </div>
          <span>{run.disagreement.toFixed(2)}</span>
        </section>
      )}

      {council && run.dissent_notes && (
        <section className="dissent">
          <h4>Dissent</h4>
          <p>{run.dissent_notes}</p>
        </section>
      )}

      <section className="answer">
        <h4>Final answer</h4>
        <p>
          <Typewriter text={run.final_answer} />
        </p>
      </section>
    </div>
  )
}
