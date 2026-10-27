"""FastAPI backend for the step-by-step viewer. Run from the project root:

    uvicorn server.main:app --reload --port 8000

The ingest endpoints read the saved index (data/index) and the PDF. The query endpoint runs the live pipeline
and returns the output of every stage: routing, dense and BM25 search, fusion, parent scoring, reranking, sources.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from contextlib import asynccontextmanager
from statistics import median

import pymupdf
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.answer import SYSTEM_PROMPT, answer, load_retriever
from app.config import ROOT, get_settings
from app.llm import Gemini
from app.retriever import RERANK_PROMPT, ROUTER_PROMPT, Retriever

settings = get_settings()
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["gemini"] = Gemini(settings)
    state["retriever"] = load_retriever(settings, state["gemini"])
    state["overview"] = build_overview(state["retriever"])
    yield


app = FastAPI(title="World Bank GEP RAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


def build_overview(retriever: Retriever) -> dict:
    """The ingest stages as data: the heading structure found in the PDF, and how the text was chunked."""
    doc = pymupdf.open(str(settings.pdf_path))
    child_count = Counter(c.parent_id for c in retriever.children)

    headings: dict[tuple, dict] = {}
    for p in retriever.parents.values():
        key = (p.chapter, p.section, p.subsection)
        row = headings.setdefault(
            key,
            {
                "path": p.path or "(before the first heading)",
                "chapter": p.chapter,
                "section": p.section,
                "subsection": p.subsection,
                "region": p.region,
                "page_start": p.page_start,
                "page_end": p.page_end,
                "parents": 0,
                "children": 0,
            },
        )
        row["page_end"] = max(row["page_end"], p.page_end)
        row["parents"] += 1
        row["children"] += child_count[p.id]

    chapters = list(dict.fromkeys(p.chapter for p in retriever.parents.values() if p.chapter))
    regions = sorted({p.region for p in retriever.parents.values() if p.region})
    sample = json.loads((ROOT / "eval" / "sample_questions.json").read_text(encoding="utf-8"))
    return {
        "pdf": {"name": settings.pdf_path.name, "pages": doc.page_count},
        "headings": list(headings.values()),
        "counts": {
            "parents": len(retriever.parents),
            "children": len(retriever.children),
            "parent_median_chars": int(median(len(p.text) for p in retriever.parents.values())),
            "child_median_chars": int(median(len(c.text) for c in retriever.children)),
            "chapters": chapters,
            "regions": regions,
        },
        "index": {
            "vectors": list(retriever.vectors.shape),
            "embedding_model": settings.embedding_model,
            "parent_max_chars": settings.parent_max_chars,
            "child_max_chars": settings.child_max_chars,
            "child_overlap_sentences": settings.child_overlap_sentences,
            "dense_k": settings.dense_k,
            "bm25_k": settings.bm25_k,
            "rrf_k": settings.rrf_k,
            "parent_candidates": settings.parent_candidates,
            "final_parents": settings.final_parents,
            "use_reranker": settings.use_reranker,
        },
        "models": {"answer": settings.llm_model, "router": settings.router_model},
        "sample_questions": [q["question"] for q in sample],
    }


@app.get("/api/health")
def health() -> dict:
    r = state["retriever"]
    return {"status": "ok", "parents": len(r.parents), "children": len(r.children), "llm_model": settings.llm_model}


@app.get("/api/overview")
def overview() -> dict:
    return state["overview"]


@app.get("/api/parents")
def parents() -> list[dict]:
    r: Retriever = state["retriever"]
    kids: dict[str, list[str]] = {}
    for c in r.children:
        kids.setdefault(c.parent_id, []).append(c.id)
    return [
        {
            "id": p.id,
            "chapter": p.chapter,
            "section": p.section,
            "subsection": p.subsection,
            "region": p.region,
            "path": p.path,
            "pages": p.pages,
            "page_start": p.page_start,
            "text": p.text,
            "child_ids": kids.get(p.id, []),
        }
        for p in r.parents.values()
    ]


@app.get("/api/children")
def children() -> list[dict]:
    return [
        {"id": c.id, "parent_id": c.parent_id, "path": c.path, "page": c.page, "region": c.region, "text": c.text}
        for c in state["retriever"].children
    ]


@app.get("/api/pages/{page}/image")
def page_image(page: int) -> Response:
    doc = pymupdf.open(str(settings.pdf_path))
    if not 1 <= page <= doc.page_count:
        raise HTTPException(404, f"The report has {doc.page_count} pages.")
    return Response(doc[page - 1].get_pixmap(dpi=110).tobytes("png"), media_type="image/png")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    rerank: bool | None = None  # None = the setting in app/config.py


@app.post("/api/query")
def query(req: QueryRequest) -> dict:
    retriever: Retriever = state["retriever"]
    try:
        res = answer(req.question, retriever, state["gemini"], settings, rerank=req.rerank)
    except RuntimeError as e:
        raise HTTPException(502, f"Model call failed: {e}") from e

    sr = res.search
    route = sr.route
    cid = {c.id: c for c in retriever.children}
    dense_rank = {cid_: r for r, (cid_, _) in enumerate(sr.dense, 1)}
    bm25_rank = {cid_: r for r, (cid_, _) in enumerate(sr.bm25, 1)}
    cited = {int(n) for group in re.findall(r"\[((?:S\d+[,\s]*)+)\]", res.text) for n in re.findall(r"S(\d+)", group)}
    timings = {**sr.timings_ms, "generation_ms": res.generation_ms}

    return {
        "question": req.question,
        "answer": res.text,
        "found": not res.refused,
        "model": settings.llm_model,
        "router_model": settings.router_model,
        "timings_ms": timings,
        "route": {
            "query": route.query,
            "chapters": sorted(route.chapters),
            "regions": sorted(route.regions),
            "source": route.source,
            "describe": route.describe(),
        },
        "router_prompt": ROUTER_PROMPT.format(regions=", ".join(retriever.region_names), question=req.question),
        "rerank_template": RERANK_PROMPT,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": res.prompt,
        "filter_size": sr.filter_size,
        "total_children": sr.total_children,
        "dense": [{"chunk_id": i, "rank": r, "score": round(s, 4)} for r, (i, s) in enumerate(sr.dense, 1)],
        "bm25": [{"chunk_id": i, "rank": r, "score": round(s, 3)} for r, (i, s) in enumerate(sr.bm25, 1)],
        "fused": [
            {
                "chunk_id": i,
                "rank": r,
                "score": round(s, 5),
                "dense_rank": dense_rank.get(i),
                "bm25_rank": bm25_rank.get(i),
                "parent_id": cid[i].parent_id,
            }
            for r, (i, s) in enumerate(sr.fused, 1)
        ],
        "parent_scores": [
            {"parent_id": pid, "rank": r, "score": round(s, 5), "path": retriever.parents[pid].path, "pages": retriever.parents[pid].pages}
            for r, (pid, s) in enumerate(sr.parent_scores, 1)
        ],
        "parent_order": sr.parent_order,
        "reranked": sr.reranked,
        "sources": [
            {
                "index": n,
                "parent_id": p.id,
                "path": p.path,
                "pages": p.pages,
                "text": p.text,
                "cited": n in cited,
            }
            for n, p in enumerate(res.sources, 1)
        ],
    }
