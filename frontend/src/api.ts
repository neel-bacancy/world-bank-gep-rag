export interface Heading {
  path: string
  chapter: string | null
  section: string | null
  subsection: string | null
  region: string | null
  page_start: number
  page_end: number
  parents: number
  children: number
}

export interface Overview {
  pdf: { name: string; pages: number }
  headings: Heading[]
  counts: {
    parents: number
    children: number
    parent_median_chars: number
    child_median_chars: number
    chapters: string[]
    regions: string[]
  }
  index: {
    vectors: number[]
    embedding_model: string
    parent_max_chars: number
    child_max_chars: number
    child_overlap_sentences: number
    dense_k: number
    bm25_k: number
    rrf_k: number
    parent_candidates: number
    final_parents: number
    use_reranker: boolean
  }
  models: { answer: string; router: string }
  sample_questions: string[]
}

export interface ParentItem {
  id: string
  chapter: string | null
  section: string | null
  subsection: string | null
  region: string | null
  path: string
  pages: string
  page_start: number
  text: string
  child_ids: string[]
}

export interface ChildItem {
  id: string
  parent_id: string
  path: string
  page: number
  region: string | null
  text: string
}

export interface Scored {
  chunk_id: string
  rank: number
  score: number
}

export interface Fused {
  chunk_id: string
  rank: number
  score: number
  dense_rank: number | null
  bm25_rank: number | null
  parent_id: string
}

export interface ParentScore {
  parent_id: string
  rank: number
  score: number
  path: string
  pages: string
}

export interface Source {
  index: number
  parent_id: string
  path: string
  pages: string
  text: string
  cited: boolean
}

export interface QueryResult {
  question: string
  answer: string
  found: boolean
  model: string
  router_model: string
  timings_ms: Record<string, number>
  route: { query: string; chapters: string[]; regions: string[]; source: string; describe: string }
  router_prompt: string
  rerank_template: string
  system_prompt: string
  user_prompt: string
  filter_size: number
  total_children: number
  dense: Scored[]
  bm25: Scored[]
  fused: Fused[]
  parent_scores: ParentScore[]
  parent_order: string[]
  reranked: boolean
  sources: Source[]
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(await errorText(res))
  return res.json() as Promise<T>
}

async function errorText(res: Response): Promise<string> {
  const body = await res.json().catch(() => null)
  return typeof body?.detail === 'string' ? body.detail : `Request failed (${res.status})`
}

export const getOverview = () => get<Overview>('/api/overview')

export const getParents = () => get<ParentItem[]>('/api/parents')

export const getChildren = () => get<ChildItem[]>('/api/children')

export const askQuestion = async (question: string, rerank: boolean): Promise<QueryResult> => {
  const res = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, rerank }),
  })
  if (!res.ok) throw new Error(await errorText(res))
  return res.json() as Promise<QueryResult>
}

export const pageImageUrl = (page: number) => `/api/pages/${page}/image`
