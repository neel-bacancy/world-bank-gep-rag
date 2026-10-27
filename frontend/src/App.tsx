import { useEffect, useMemo, useState } from 'react'
import { getChildren, getOverview, getParents, type ChildItem, type Overview, type ParentItem } from './api'
import { IngestView } from './components/IngestView'
import { QueryView } from './components/QueryView'
import './App.css'

type Tab = 'ask' | 'ingest'

function App() {
  const [tab, setTab] = useState<Tab>('ask')
  const [overview, setOverview] = useState<Overview | null>(null)
  const [parents, setParents] = useState<ParentItem[]>([])
  const [children, setChildren] = useState<ChildItem[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([getOverview(), getParents(), getChildren()])
      .then(([o, p, c]) => {
        setOverview(o)
        setParents(p)
        setChildren(c)
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  const childById = useMemo(() => new Map(children.map((c) => [c.id, c])), [children])

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>World Bank Global Economic Prospects: RAG</h1>
          <p className="sub">
            Every step from the PDF to the answer, shown as it runs. Answers come from the Global Economic Prospects, June 2026 report, with chapter,
            section and page citations.
          </p>
        </div>
        <div className={`status ${overview ? 'ok' : error ? 'down' : ''}`}>
          {overview
            ? `${overview.counts.parents} sections · ${overview.counts.children} chunks · ${overview.models.answer}`
            : error
              ? 'Backend offline'
              : 'Connecting…'}
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === 'ask' ? 'active' : ''} onClick={() => setTab('ask')}>
          Ask a question
        </button>
        <button className={tab === 'ingest' ? 'active' : ''} onClick={() => setTab('ingest')}>
          How the index was built
        </button>
      </nav>

      {error && (
        <div className="error">
          {error}. Start the backend from the project root: <code>uvicorn server.main:app --port 8000</code>
        </div>
      )}

      <div hidden={tab !== 'ask'}>
        <QueryView overview={overview} childById={childById} />
      </div>
      <div hidden={tab !== 'ingest'}>
        <IngestView overview={overview} parents={parents} childById={childById} />
      </div>
    </div>
  )
}

export default App
