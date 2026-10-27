"""Query routing, hybrid search (dense + BM25 fused with Reciprocal Rank Fusion), parent resolution,
and an optional LLM reranker.

Pipeline for one question:
  1. Route:   explicit mentions ("Chapter 3", "South Asia", "Executive Summary") become hard filters.
              A small LLM rewrites the question for search and maps countries to their region
              (Pakistan -> South Asia). Explicit mentions win over the LLM.
  2. Search:  dense cosine over the children and BM25 over the children, inside the filter.
  3. Fuse:    RRF, score(d) = sum over lists of 1 / (k + rank), k = 60.
  4. Resolve: children -> their parents (the full section), scored by the sum of child RRF scores.
  5. Rerank:  optional LLM listwise ranking of the top parents. Measured in eval, not assumed.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.chunker import Child, Parent
from app.config import Settings
from app.embedder import Embedder
from app.llm import Gemini

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was what which who will with "
    "does do say says report about main".split()
)
# Words that appear on almost every page of this report. Left in, they match the title and the bibliography.
DOCUMENT_TERMS = frozenset("world bank economic prospects june 2026 gep".split())
TOKEN = re.compile(r"[a-z0-9]+")

ROUTER_PROMPT = """You prepare a question about the World Bank report "Global Economic Prospects, June 2026" for search.

Regional sections of the report: {regions}

Return JSON with two keys:
- "query": a standalone search query that keeps the key terms of the question. Do not add the report's
  title, its year, or the words "report", "World Bank" or "Global Economic Prospects".
- "regions": the regional sections the question is about, copied exactly from the list above.
  Map countries to their region (for example Pakistan -> South Asia, Nigeria -> Sub-Saharan Africa).
  Return [] for global questions or when no region is implied.

Question: {question}"""

RERANK_PROMPT = """Question: {question}

Passages from the report:

{passages}

Rank every passage id from most to least useful for answering the question. Passages about the exact region,
chapter or topic the question names come first. Return JSON: {{"ranking": ["P1", "P2", ...]}}"""


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and t not in DOCUMENT_TERMS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if len(docs) else 1.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.dl / self.avgdl)
        for term in set(query):
            if term not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term)
                if f:
                    out[i] += self.idf[term] * f * (self.k1 + 1) / (f + norm[i])
        return out


@dataclass
class Route:
    query: str
    chapters: set[str] = field(default_factory=set)
    regions: set[str] = field(default_factory=set)
    source: str = "none"  # "rule", "router" or "none"

    def describe(self) -> str:
        parts = []
        if self.chapters:
            parts.append("chapter=" + ", ".join(sorted(self.chapters)))
        if self.regions:
            parts.append("region=" + ", ".join(sorted(self.regions)))
        return "; ".join(parts) or "no filter (whole report)"


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


@dataclass
class SearchResult:
    route: Route
    parents: list[Parent]
    children: list[tuple[Child, float]]  # the child hits that produced the parents, for inspection
    # Trace of each stage, for the step-by-step view. Child ids and parent ids, not objects.
    filter_size: int = 0  # children inside the route's filter (the search space)
    total_children: int = 0
    dense: list[tuple[str, float]] = field(default_factory=list)  # (child id, cosine), best first
    bm25: list[tuple[str, float]] = field(default_factory=list)  # (child id, BM25), best first
    fused: list[tuple[str, float]] = field(default_factory=list)  # (child id, RRF), best first
    parent_scores: list[tuple[str, float]] = field(default_factory=list)  # (parent id, summed RRF), before rerank
    parent_order: list[str] = field(default_factory=list)  # parent ids after reranking (= score order if off)
    reranked: bool = False
    timings_ms: dict[str, int] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        settings: Settings,
        gemini: Gemini,
        embedder: Embedder,
        parents: list[Parent],
        children: list[Child],
        vectors: np.ndarray,
    ):
        self.s = settings
        self.gem = gemini
        self.embedder = embedder
        self.parents = {p.id: p for p in parents}
        self.children = children
        self.vectors = vectors
        self.bm25 = BM25([tokens(c.embed_text) for c in children])
        self.region_names = sorted({p.section for p in parents if p.region and p.section})
        self.chapter_labels = sorted({p.chapter for p in parents if p.chapter})
        # Front-matter labels ("Executive Summary", "Foreword") are chapter-level labels that are not numbered.
        self.front_labels = sorted(
            {
                label
                for label in self.chapter_labels
                if not label.startswith("Chapter") and "Economic Prospects" not in label and "JUNE" not in label.upper()
            }
        )

    # ---- routing ---------------------------------------------------------------------------
    def _explicit_route(self, question: str, route: Route) -> None:
        number = re.search(r"\bchapter\s+(\d)\b", question, re.I)
        if number:
            route.chapters = {label for label in self.chapter_labels if label.startswith(f"Chapter {number.group(1)}:")}
        for label in self.front_labels:
            if label.lower() in question.lower():
                route.chapters = {label}
        for region in self.region_names:
            if re.search(rf"\b{re.escape(region)}\b", question, re.I):
                route.regions.add(region)
        if route.chapters or route.regions:
            route.source = "rule"

    def route(self, question: str) -> Route:
        route = Route(query=question)
        self._explicit_route(question, route)
        prompt = ROUTER_PROMPT.format(regions=", ".join(self.region_names), question=question)
        try:
            data = json.loads(self.gem.generate(prompt, model=self.s.router_model, json_output=True))
            route.query = (data.get("query") or question).strip()
            if not route.regions and not route.chapters:
                inferred = {r for r in data.get("regions", []) if r in self.region_names}
                if inferred:
                    route.regions = inferred
                    route.source = "router"
        except (json.JSONDecodeError, RuntimeError, AttributeError):
            pass  # routing is an improvement, not a requirement: fall back to the raw question
        return route

    # ---- search ----------------------------------------------------------------------------
    def _candidate_mask(self, route: Route) -> np.ndarray:
        mask = np.ones(len(self.children), dtype=bool)
        if route.chapters:
            mask &= np.array([c.chapter in route.chapters for c in self.children])
        if route.regions:
            mask &= np.array([c.region in route.regions for c in self.children])
        if not mask.any():  # a filter that matches nothing must not return nothing
            return np.ones(len(self.children), dtype=bool)
        return mask

    def search(self, question: str, rerank: bool | None = None) -> SearchResult:
        rerank = self.s.use_reranker if rerank is None else rerank
        timings: dict[str, int] = {}

        t = time.perf_counter()
        route = self.route(question)
        timings["route_ms"] = elapsed_ms(t)
        idx = np.flatnonzero(self._candidate_mask(route))

        t = time.perf_counter()
        query_vec = self.embedder.embed([route.query], task="RETRIEVAL_QUERY")[0]
        timings["embed_ms"] = elapsed_ms(t)

        t = time.perf_counter()
        dense = self.vectors[idx] @ query_vec
        order_d = np.argsort(-dense)[: self.s.dense_k]
        dense_top = idx[order_d]

        bm25_all = self.bm25.scores(tokens(f"{question} {route.query}"))
        bm25_scores = bm25_all[idx]
        order = np.argsort(-bm25_scores)[: self.s.bm25_k]
        order = order[bm25_scores[order] > 0]
        bm25_top = idx[order]

        fused: defaultdict[int, float] = defaultdict(float)
        for rank, i in enumerate(dense_top):
            fused[int(i)] += 1 / (self.s.rrf_k + rank + 1)
        for rank, i in enumerate(bm25_top):
            fused[int(i)] += 1 / (self.s.rrf_k + rank + 1)
        ranked_children = sorted(fused, key=lambda i: -fused[i])

        parent_score: defaultdict[str, float] = defaultdict(float)
        for i in ranked_children[: self.s.dense_k]:
            parent_score[self.children[i].parent_id] += fused[i]
        parent_ids = sorted(parent_score, key=lambda pid: -parent_score[pid])[: self.s.parent_candidates]
        timings["search_ms"] = elapsed_ms(t)

        t = time.perf_counter()
        reranked = bool(rerank and len(parent_ids) > 1)
        final_order = self._rerank(question, parent_ids) if reranked else parent_ids
        timings["rerank_ms"] = elapsed_ms(t) if reranked else 0

        hits = [(self.children[i], fused[i]) for i in ranked_children[: self.s.dense_k]]
        return SearchResult(
            route=route,
            parents=[self.parents[pid] for pid in final_order[: self.s.final_parents]],
            children=hits,
            filter_size=len(idx),
            total_children=len(self.children),
            dense=[(self.children[int(i)].id, float(dense[j])) for j, i in zip(order_d, dense_top)],
            bm25=[(self.children[int(i)].id, float(bm25_scores[j])) for j, i in zip(order, bm25_top)],
            fused=[(self.children[i].id, fused[i]) for i in ranked_children[: self.s.dense_k]],
            parent_scores=[(pid, parent_score[pid]) for pid in parent_ids],
            parent_order=final_order,
            reranked=reranked,
            timings_ms=timings,
        )

    def _rerank(self, question: str, parent_ids: list[str]) -> list[str]:
        """One listwise call to the small model. Falls back to fused order if the reply can't be parsed."""
        passages = "\n\n".join(
            f"[P{n}] {self.parents[pid].path}\n{self.parents[pid].text[:1500]}" for n, pid in enumerate(parent_ids, 1)
        )
        prompt = RERANK_PROMPT.format(question=question, passages=passages)
        try:
            data = json.loads(self.gem.generate(prompt, model=self.s.router_model, json_output=True))
            ranked = [parent_ids[int(x[1:]) - 1] for x in data["ranking"] if x[1:].isdigit() and 0 < int(x[1:]) <= len(parent_ids)]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError):
            return parent_ids
        seen = list(dict.fromkeys(ranked))
        return seen + [pid for pid in parent_ids if pid not in seen]
