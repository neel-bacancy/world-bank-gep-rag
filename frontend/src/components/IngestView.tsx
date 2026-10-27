import { useState, type ReactNode } from 'react'
import { pageImageUrl, type ChildItem, type Heading, type Overview, type ParentItem } from '../api'
import { Step } from './Step'

interface Props {
  overview: Overview | null
  parents: ParentItem[]
  childById: Map<string, ChildItem>
}

export function IngestView({ overview, parents, childById }: Props) {
  const [page, setPage] = useState(1)
  const [chapter, setChapter] = useState('all')
  const [search, setSearch] = useState('')

  if (!overview) return <p className="empty">Loading the ingest overview…</p>

  const { counts, index } = overview
  const onPage = parents.filter((p) => p.page_start === page)
  const needle = search.trim().toLowerCase()
  const shown = parents.filter(
    (p) => (chapter === 'all' || p.chapter === chapter) && (!needle || p.text.toLowerCase().includes(needle) || p.path.toLowerCase().includes(needle)),
  )

  return (
    <div className="ingest">
      <p className="intro">
        Ingestion runs once with <code>python -m scripts.ingest</code>. This view shows what that run produced, stage by stage. It reads the
        saved index and the PDF, so it makes no model calls.
      </p>

      <Step
        n={1}
        title="Read the report"
        what={`${overview.pdf.name} has ${overview.pdf.pages} pages. Text is read in reading order (two columns, with running headers removed by position). Pick a page to see it and the sections that start on it.`}
      >
        <div className="thumbs">
          {Array.from({ length: overview.pdf.pages }, (_, i) => i + 1).map((n) => (
            <button key={n} className={`thumb ${n === page ? 'active' : ''}`} onClick={() => setPage(n)}>
              <img src={pageImageUrl(n)} alt="" loading="lazy" />
              <span>p. {n}</span>
            </button>
          ))}
        </div>
        <div className="page-detail">
          <img className="page-img" src={pageImageUrl(page)} alt={`Page ${page}`} />
          <div className="page-facts">
            <h4>Sections that start on p. {page}</h4>
            {onPage.length === 0 ? (
              <p className="muted">No new section starts on this page. Its text belongs to a section begun earlier.</p>
            ) : (
              onPage.map((p) => (
                <div key={p.id} className="heading-hit">
                  <strong>{p.path}</strong>
                  <span className="muted">
                    {' '}
                    {p.id} · {p.child_ids.length} chunks
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </Step>

      <Step
        n={2}
        title="Structure from the fonts"
        what="Headings are read from font size and weight, not guessed from the text: chapter labels and titles, section headings (bold, 15.5pt or larger), subsections (bold, smaller). Front and back matter get their own labels. Each row is one heading path, with its pages and how much text sits under it."
      >
        <div className="stats">
          <div>
            <span className="muted">Chapters</span>
            <strong>{counts.chapters.length}</strong>
          </div>
          <div>
            <span className="muted">Heading paths</span>
            <strong>{overview.headings.length}</strong>
          </div>
          <div>
            <span className="muted">Regional sections</span>
            <strong>{counts.regions.length}</strong>
          </div>
        </div>
        <div className="headings">
          {overview.headings.map((h) => (
            <HeadingRow key={h.path} h={h} />
          ))}
        </div>
      </Step>

      <Step
        n={3}
        title="Chunk: sections and sentence windows"
        what={`Each heading path's text is packed into parents of up to ${index.parent_max_chars} characters. The answer model reads parents, so it gets the whole section. Children are sentence windows of up to ${index.child_max_chars} characters, with ${index.child_overlap_sentences} sentence of overlap, and they are what gets searched. Each child points back to its parent.`}
      >
        <div className="stats">
          <div>
            <span className="muted">Parents</span>
            <strong>{counts.parents}</strong>
          </div>
          <div>
            <span className="muted">Children</span>
            <strong>{counts.children}</strong>
          </div>
          <div>
            <span className="muted">Median parent</span>
            <strong>{counts.parent_median_chars} chars</strong>
          </div>
          <div>
            <span className="muted">Median child</span>
            <strong>{counts.child_median_chars} chars</strong>
          </div>
        </div>

        <div className="filters">
          <select value={chapter} onChange={(e) => setChapter(e.target.value)}>
            <option value="all">All chapters</option>
            {counts.chapters.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <input type="search" placeholder="Filter by heading or text…" value={search} onChange={(e) => setSearch(e.target.value)} />
          <span className="muted">
            {shown.length} of {parents.length}
          </span>
        </div>
        <div className="chunk-list">
          {shown.map((p) => (
            <details key={p.id} className="chunk">
              <summary>
                <code>{p.id}</code>
                <span className="preview">{p.path}</span>
                <span className="muted">{p.pages}</span>
                <span className="muted">{p.child_ids.length} chunks</span>
              </summary>
              <pre>{p.text}</pre>
              <div className="prompt-label">Children of this section (searched individually)</div>
              {p.child_ids.map((cid) => {
                const c = childById.get(cid)
                return (
                  <div key={cid} className="child">
                    <code>{cid}</code>
                    <span className="muted">p. {c?.page}</span>
                    <pre>{c?.text}</pre>
                  </div>
                )
              })}
            </details>
          ))}
        </div>
      </Step>

      <Step
        n={4}
        title="Embed and index"
        what="Each child is embedded once with a local model, with its heading path in front of the text, so a region named in the question matches even when the body does not repeat it. Child tokens also go into a BM25 index. Queries search both, inside the route's filter."
      >
        <div className="stats">
          <div>
            <span className="muted">Dense vectors</span>
            <strong>
              {index.vectors[0]} × {index.vectors[1]}
            </strong>
          </div>
          <div>
            <span className="muted">Embedding model</span>
            <strong>{index.embedding_model}</strong>
          </div>
        </div>
        <table className="grid narrow">
          <tbody>
            <Row label="Dense candidates (top k)" value={index.dense_k} />
            <Row label="BM25 candidates (top k)" value={index.bm25_k} />
            <Row label="RRF constant k" value={index.rrf_k} />
            <Row label="Parent candidates for reranking" value={index.parent_candidates} />
            <Row label="Parents to answer model" value={index.final_parents} />
            <Row label="Reranker" value={index.use_reranker ? 'on' : 'off'} />
            <Row label="Answer model" value={overview.models.answer} />
            <Row label="Router model (rewrite, route, rerank)" value={overview.models.router} />
          </tbody>
        </table>
        <div className="prompt-label">Regional sections the router can pick from</div>
        <div className="chips-static">
          {counts.regions.map((g) => (
            <span key={g} className="tag">
              {g}
            </span>
          ))}
        </div>
      </Step>
    </div>
  )
}

function HeadingRow({ h }: { h: Heading }) {
  const depth = [h.chapter, h.section, h.subsection].filter(Boolean).length - 1
  const name = h.subsection ?? h.section ?? h.chapter ?? h.path
  return (
    <div className={`heading-row depth-${Math.max(depth, 0)}`} style={{ paddingLeft: `${Math.max(depth, 0) * 1.1}rem` }}>
      <span className="heading-name">{name}</span>
      {h.region && <span className="tag">region</span>}
      <span className="muted">
        pp. {h.page_start}–{h.page_end}
      </span>
      <span className="muted">
        {h.parents} parent{h.parents === 1 ? '' : 's'} · {h.children} chunk{h.children === 1 ? '' : 's'}
      </span>
    </div>
  )
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <tr>
      <td>{label}</td>
      <td>
        <strong>{value}</strong>
      </td>
    </tr>
  )
}
