import { useState, type FormEvent, type ReactNode } from 'react'
import { askQuestion, type ChildItem, type Overview, type QueryResult, type Scored } from '../api'
import { AnswerText } from './AnswerText'
import { Step } from './Step'

interface Props {
  overview: Overview | null
  childById: Map<string, ChildItem>
}

export function QueryView({ overview, childById }: Props) {
  const [question, setQuestion] = useState('')
  const [rerankChoice, setRerankChoice] = useState<boolean | null>(null)
  const [result, setResult] = useState<QueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const rerank = rerankChoice ?? overview?.index.use_reranker ?? false

  const ask = async (text: string) => {
    const q = text.trim()
    if (q.length < 3 || loading) return
    setQuestion(q)
    setLoading(true)
    setError(null)
    try {
      setResult(await askQuestion(q, rerank))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    ask(question)
  }

  const cite = (n: number) => document.getElementById(`src-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })

  return (
    <div className="query">
      <form className="ask" onSubmit={onSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              ask(question)
            }
          }}
          placeholder="Ask about growth, inflation, debt or a region named in the report…"
          rows={2}
          disabled={loading}
        />
        <div className="ask-row">
          <label className="toggle">
            <input type="checkbox" checked={rerank} onChange={(e) => setRerankChoice(e.target.checked)} disabled={loading} />
            Use the reranker (one extra model call)
          </label>
          <button type="submit" disabled={loading || question.trim().length < 3}>
            {loading ? 'Running…' : 'Ask'}
          </button>
        </div>
      </form>

      <div className="samples">
        {overview?.sample_questions.map((q) => (
          <button key={q} type="button" className="chip" onClick={() => ask(q)} disabled={loading}>
            {q}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}
      {loading && <div className="running">Running the pipeline: route, search, fuse, rank sections, then answer…</div>}
      {!result && !loading && !error && <p className="empty">Ask a question, or pick a sample, to see each step of the pipeline.</p>}

      {result && (
        <Results r={result} overview={overview} childById={childById} onCite={cite} />
      )}
    </div>
  )
}

function Results({
  r,
  overview,
  childById,
  onCite,
}: {
  r: QueryResult
  overview: Overview | null
  childById: Map<string, ChildItem>
  onCite: (n: number) => void
}) {
  const t = r.timings_ms
  const total = Object.values(t).reduce((a, b) => a + b, 0)
  const denseMax = r.dense[0]?.score ?? 1
  const bm25Max = r.bm25[0]?.score ?? 1
  const parentMax = r.parent_scores[0]?.score ?? 1
  const finalN = overview?.index.final_parents ?? r.sources.length
  const beforeRerank = new Map(r.parent_scores.map((p) => [p.parent_id, p.rank]))
  const afterRerank = new Map(r.parent_order.map((id, i) => [id, i + 1]))
  const childPath = (id: string) => childById.get(id)?.path ?? id
  const childPage = (id: string) => childById.get(id)?.page

  const routeBits = [
    ...r.route.chapters.map((c) => `chapter: ${c}`),
    ...r.route.regions.map((g) => `region: ${g}`),
  ]

  return (
    <div className="results">
      <div className="summary">
        <span>
          Total <strong>{(total / 1000).toFixed(1)} s</strong>
        </span>
        <span>route {t.route_ms} ms</span>
        <span>embed {t.embed_ms} ms</span>
        <span>search {t.search_ms} ms</span>
        <span>rerank {t.rerank_ms} ms</span>
        <span>generation {t.generation_ms} ms</span>
      </div>

      <Step n={1} title="Your question" what="The question as it was sent to the pipeline.">
        <blockquote className="question">{r.question}</blockquote>
      </Step>

      <Step
        n={2}
        title="Route the question"
        what="Explicit mentions (a chapter number, a front-matter title, a region name) become hard filters. A small model rewrites the question for search and maps countries to regions. An explicit mention wins over the model."
        timing={t.route_ms}
      >
        <div className="kv">
          <span>Search query</span>
          <code>{r.route.query}</code>
        </div>
        <div className="kv">
          <span>Filter</span>
          <span className={`route-source ${r.route.source}`}>{r.route.describe}</span>
          <span className="muted">from {r.route.source === 'none' ? 'no rule or model match' : r.route.source}</span>
        </div>
        <div className="kv">
          <span>Search space</span>
          <strong>
            {r.filter_size} of {r.total_children} chunks
          </strong>
          {routeBits.length === 0 && <span className="muted">(whole report: the filter matched nothing, or none was implied)</span>}
        </div>
        <Details label="Router prompt">
          <pre>{r.router_prompt}</pre>
        </Details>
      </Step>

      <Step
        n={3}
        title="Dense search: meaning"
        what={`The query is embedded with ${overview?.index.embedding_model ?? 'the local embedding model'} and compared with every chunk in the search space by cosine similarity. The top ${r.dense.length} are kept. Bars are relative to the top result.`}
        timing={t.embed_ms}
      >
        <ScoreList rows={r.dense} max={denseMax} decimals={3} label={childPath} page={childPage} />
      </Step>

      <Step n={4} title="BM25 search: keywords" what="Keyword ranking over the same chunks, using the question and the search query. Only chunks with a nonzero score appear.">
        <ScoreList rows={r.bm25} max={bm25Max} decimals={2} label={childPath} page={childPage} />
      </Step>

      <Step
        n={5}
        title="Fuse the two lists"
        what={`Reciprocal Rank Fusion: each chunk gets 1 / (60 + rank) from every list it appears in. Chunks found by both searches are highlighted. The top ${r.fused.length} are kept for the next step.`}
      >
        <table className="grid">
          <thead>
            <tr>
              <th>#</th>
              <th>Chunk</th>
              <th>Section</th>
              <th>Dense</th>
              <th>BM25</th>
              <th>RRF</th>
            </tr>
          </thead>
          <tbody>
            {r.fused.map((f) => (
              <tr key={f.chunk_id} className={f.dense_rank && f.bm25_rank ? 'both' : ''}>
                <td>{f.rank}</td>
                <td>
                  <code>{f.chunk_id}</code>
                </td>
                <td className="path">{childPath(f.chunk_id)}</td>
                <td>{f.dense_rank ?? '—'}</td>
                <td>{f.bm25_rank ?? '—'}</td>
                <td>{f.score.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Step>

      <Step
        n={6}
        title="Vote for sections"
        what={`Each chunk resolves to its parent: the whole section it sits in. A section's score is the sum of its chunks' RRF scores. The top ${r.parent_scores.length} sections are candidates.`}
      >
        <ol className="score-list parents">
          {r.parent_scores.map((p) => (
            <li key={p.parent_id}>
              <span className="rank">{p.rank}</span>
              <code>{p.parent_id}</code>
              <span className="path" title={p.path}>
                {p.path} <span className="muted">({p.pages})</span>
              </span>
              <span className="bar-track">
                <span className="bar-fill" style={{ width: `${(p.score / parentMax) * 100}%` }} />
              </span>
              <span className="score">{p.score.toFixed(4)}</span>
            </li>
          ))}
        </ol>
      </Step>

      <Step
        n={7}
        title="Rerank the sections"
        what={
          r.reranked
            ? `A small model reads the candidate sections and orders them by how well they answer the question. The top ${finalN} go to the answer model.`
            : `The reranker is off for this question, so the vote order is used as it is. The top ${finalN} go to the answer model.`
        }
        timing={r.reranked ? t.rerank_ms : undefined}
      >
        {r.reranked && (
          <table className="grid">
            <thead>
              <tr>
                <th>Section</th>
                <th>Vote rank</th>
                <th>After rerank</th>
                <th>Move</th>
              </tr>
            </thead>
            <tbody>
              {r.parent_scores.map((p) => {
                const before = beforeRerank.get(p.parent_id) ?? p.rank
                const after = afterRerank.get(p.parent_id) ?? before
                const move = after < before ? `↑ ${before - after}` : after > before ? `↓ ${after - before}` : '='
                return (
                  <tr key={p.parent_id} className={after <= finalN ? 'kept' : ''}>
                    <td className="path">{p.path}</td>
                    <td>{before}</td>
                    <td>{after}</td>
                    <td>{move}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
        {r.reranked && (
          <Details label="Reranker instruction">
            <pre>{r.rerank_template}</pre>
          </Details>
        )}
      </Step>

      <Step
        n={8}
        title="Context for the answer model"
        what={`The top ${r.sources.length} sections, in order, are numbered S1 to S${r.sources.length} and sent to ${r.model} with the question. Each source is the full section, not just the matching chunk. Sections the answer cites are marked.`}
      >
        <div className="sources">
          {r.sources.map((s) => (
            <article key={s.index} id={`src-${s.index}`} className={`source ${s.cited ? 'cited' : ''}`}>
              <header>
                <span className="tag-s">S{s.index}</span>
                <span className="path" title={s.path}>
                  {s.path}
                </span>
                <span className="muted">{s.pages}</span>
                {s.cited && <span className="badge">cited</span>}
              </header>
              <pre>{s.text}</pre>
            </article>
          ))}
        </div>
        <Details label="Full prompt sent to the model">
          <div className="prompt-label">System prompt</div>
          <pre>{r.system_prompt}</pre>
          <div className="prompt-label">User message</div>
          <pre>{r.user_prompt}</pre>
        </Details>
      </Step>

      <Step n={9} title="Answer" what="The model writes the answer from the sections only. Click a citation to jump to its source." timing={t.generation_ms}>
        {!r.found && <div className="notice">The model reports that the report does not contain this information.</div>}
        <AnswerText text={r.answer} onCite={onCite} />
        <div className="stats">
          <div>
            <span className="muted">Sections cited</span>
            <strong>{r.sources.filter((s) => s.cited).length}</strong>
          </div>
          <div>
            <span className="muted">Model</span>
            <strong>{r.model}</strong>
          </div>
        </div>
      </Step>
    </div>
  )
}

function ScoreList({
  rows,
  max,
  decimals,
  label,
  page,
}: {
  rows: Scored[]
  max: number
  decimals: number
  label: (id: string) => string
  page: (id: string) => number | undefined
}) {
  return (
    <ol className="score-list">
      {rows.map((row) => (
        <li key={row.chunk_id}>
          <span className="rank">{row.rank}</span>
          <code>{row.chunk_id}</code>
          <span className="path" title={label(row.chunk_id)}>
            {label(row.chunk_id)}
            {page(row.chunk_id) !== undefined && <span className="muted"> · p. {page(row.chunk_id)}</span>}
          </span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${Math.max(0, (row.score / max) * 100)}%` }} />
          </span>
          <span className="score">{row.score.toFixed(decimals)}</span>
        </li>
      ))}
    </ol>
  )
}

function Details({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="details">
      <summary>{label}</summary>
      {children}
    </details>
  )
}
