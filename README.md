# World Bank Global Economic Prospects (June 2026): Structure-Aware RAG

Question answering over the 200-page *Global Economic Prospects, June 2026* report. The point of this project is
structure: the system must find the right chapter and section, and must not mix one region's facts into another's.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # then set GEMINI_API_KEY
python -m scripts.ingest        # parse by structure, chunk, embed locally, save to data/index
python -m scripts.ask "What is the growth outlook for South Asia in this report?"
python -m eval.run_eval --compare
```

Gemini does the generation, query rewriting and optional reranking. Embeddings run locally (BAAI bge-small, via
fastembed), so ingesting and searching do not use the Gemini embedding quota. The model downloads once, on first use.

## How it works

### 1. Structure-aware parsing (`app/parser.py`)
Headings are read from the fonts, not guessed from the text:

| Element | Detection |
|---|---|
| Chapter | `CHAPTER n` label (26pt) plus the 30pt title. They are paired per page, because the title can sit above the label in reading order. |
| Front and back matter | Other 26pt titles: Executive Summary, Foreword, Statistical Appendix |
| Section | Bold, 15.5pt or larger: "South Asia", "Global context", "Policy challenges" |
| Subsection | Bold 13.5 to 15.5pt, or a short bold-italic line: "Downside risks", "Outlook" |
| Boxed article | `BOX n.n Title` kept separate from the surrounding prose |
| Bibliography | The `References` sections are left out of the index. They list other works and contain no findings. |

Other steps:
- Two-column reading order: the left column, then the right, within each band between full-width blocks.
- Running headers and footers are removed by position. Contents pages are skipped (dot leaders).
- Text repair: ligatures normalised ("ﬁ" to "fi"), words hyphenated across a line break rejoined, and the dropped "h"
  in drop caps repaired ("Te global outlook" to "The global outlook").

Result: 4,374 paragraphs across the four numbered chapters, the front and back matter, and the statistical appendix,
each with its heading path and page.

### 2. Hierarchical chunking (`app/chunker.py`)
- **Parent** = one heading path's text, packed to about 900 tokens. The answer model reads parents, so it gets the
  whole section.
- **Child** = a sentence window of about 280 tokens inside a parent, with one sentence of overlap. Children are searched.
- Each child stores its `parent_id`. A child hit resolves up to its parent.
- The embedded text starts with the heading path (`Chapter 2 > South Asia > Outlook`), so a region named in the
  question matches even when the body does not repeat it. The stored text is unchanged, so citations stay clean.
- Result: 243 parents and 806 children.

### 3. Routing (`app/retriever.py`)
- **Rules first:** "Chapter 3", "Executive Summary" and region names in the question become hard filters on the
  chapter or region. An explicit mention wins over the LLM.
- **Small LLM second** (`gemini-3.5-flash-lite`): rewrites the question for search, and maps countries to regions
  (Pakistan to South Asia). It may only pick from the real regional sections.
- A filter that would match nothing falls back to the whole report.

### 4. Hybrid retrieval
- Dense cosine (bge-small, with its query instruction prefix) and BM25 over the children, each top 30.
- Fused with Reciprocal Rank Fusion, k = 60. Children resolve to parents, scored by the sum of their child scores.
- BM25 ignores words that appear on nearly every page (the report title, "world bank", "2026"). Left in, they matched
  the bibliography and pushed the right sections out of the top results.
- Optional listwise LLM reranker over the top 8 parents. Off by default; see the evaluation.

### 5. Answering (`app/answer.py`)
Numbered source blocks, each labelled with its heading path and pages. The prompt requires `[S#]` citations, the
chapter and section named in the first sentence, figures copied exactly, and a fixed refusal sentence when the
sources don't contain the answer.

## Evaluation (`eval/`)

`sample_questions.json` has the three sample questions from the brief, plus two distractors (Sub-Saharan Africa must
not get South Asia's 6.3%; India must route to South Asia) and one out-of-scope refusal. Each one has a `note` with
the page where the answer is stated, checked against the PDF.

Latest run (`eval/results/20261026-131501.md`):

| Reranker | Passed | Retrieval hit | Answer pass |
|---|---|---|---|
| off (default) | **6/6** | 100% | 100% |
| on | 5/6 | 100% | 83% |

The reranker moved one answer on this set (q02). In the reranked run, the top parent was the chapter opener and the
Annex and Box pages outranked the Introduction, which holds the "over 70 percent" figure. Earlier, with a different
embedding model and before the bibliography fix, the reranker had been the one that helped q02. Since its effect has
flipped with the setup, it stays off by default and can be turned on with `use_reranker` in `app/config.py`.

**Changes made during evaluation, disclosed:**
1. Embeddings were first Gemini's `gemini-embedding-001`. The daily free-tier quota ran out mid-project. They were
   moved to local bge-small, which the team had discussed. Earlier results files (`20261026-124634.md`) are from the
   Gemini setup.
2. The bibliography was excluded from the index, and the report's title words were removed from BM25. Both were
   found by inspecting a failing retrieval: the rewritten query said "World Bank Global Economic Prospects" and
   matched the References pages.
3. Eval checks became regular expressions, so "4 percent" also accepts "4.0 percent" (the report's own answer is
   written both ways). The check is still strict about the figure: the number has to be there.

**Caveat:** six hand-checked questions is a smoke test, not a benchmark. The brief's pass bar is "right chapter and
section, no mixing". The eval checks the chapter and section of the retrieved parents and the presence of key
figures. It does not score every sentence of every answer.

## Limitations and next steps
- **Reranker:** the brief asked for a cross-encoder. The option here is an LLM listwise reranker, because a local
  cross-encoder would add a large model download. A cross-encoder (for example a `sentence-transformers` MiniLM)
  would replace `Retriever._rerank`. The measured effect on this set is small and changes with the setup.
- **Heading rules are tuned to this report's fonts.** The principles carry over to other reports; the thresholds in
  `app/parser.py` do not.
- **Charts and tables:** text inside charts is not extracted, and tables are read as text in reading order. The
  iMac project covers tables and charts with a vision pass.
- **Drop-cap repair** covers "Te" and "te" only. Other dropped-letter cases would need a dictionary check.
- **Eval set size:** expand to 20 or more hand-checked questions before drawing conclusions about small changes.

## Layout

```
app/
  parser.py     font-based structure, two-column order, text repair
  chunker.py    parent/child chunks with heading paths
  embedder.py   local bge-small embeddings
  retriever.py  routing, BM25 + dense, RRF, parent resolution, optional rerank
  answer.py     index loading and grounded answers
  llm.py        Gemini generation with model fallback
server/        FastAPI backend for the viewer (server/main.py)
frontend/      React + Vite viewer: ingest stages and per-question steps
scripts/        ingest.py, ask.py
eval/           sample_questions.json, run_eval.py, results/
data/raw/       GEP-Jun-2026.pdf
```

## Step-by-step viewer

A browser view of the same pipeline. The "How the index was built" tab shows the PDF pages, the heading structure
found in the fonts, the parent and child chunks, and the index settings. The "Ask a question" tab shows each
retrieval stage for one question: routing, dense and BM25 search, fusion, the section vote, reranking, the context
sent to the model, and the answer.

```bash
uvicorn server.main:app --port 8000        # backend (from this folder, with the venv active)
cd frontend && npm install && npm run dev  # frontend on http://localhost:5173
```

The viewer reads the saved index (`data/index`) and the PDF, so run ingest first. It does not re-run ingestion.
Each question makes the same Gemini calls as `scripts.ask` (router, optional rerank, answer).

