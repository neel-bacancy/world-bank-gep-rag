"""Load the saved index, then answer from the retrieved parents with numbered, cited sources."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from app.chunker import Child, Parent
from app.config import Settings
from app.embedder import Embedder
from app.llm import Gemini
from app.retriever import Retriever, SearchResult, elapsed_ms

REFUSAL = "The report does not contain this information."

SYSTEM_PROMPT = f"""You answer questions about the World Bank's Global Economic Prospects, June 2026.

Rules:
- Use only the numbered sources. Do not use outside knowledge.
- End every factual sentence with its source tag, for example [S1].
- Keep figures for different regions, countries and chapters separate. Never attribute a figure from one
  source's region or chapter to another. Copy numbers exactly as written.
- In the first sentence, name the chapter and section the answer comes from.
- If the sources do not answer the question, reply exactly: "{REFUSAL}"
- Be concise: a short paragraph or a few bullet points."""


@dataclass
class Answer:
    question: str
    text: str
    search: SearchResult
    sources: list[Parent]
    prompt: str = ""  # the user message sent to the answer model (the system prompt is SYSTEM_PROMPT)
    generation_ms: int = 0

    @property
    def refused(self) -> bool:
        return REFUSAL.lower() in self.text.lower()


def load_retriever(settings: Settings, gemini: Gemini) -> Retriever:
    payload = json.loads((settings.index_dir / "index.json").read_text(encoding="utf-8"))
    parents = [Parent(**p) for p in payload["parents"]]
    children = [Child(**c) for c in payload["children"]]
    vectors = np.load(settings.index_dir / "vectors.npy")
    return Retriever(settings, gemini, Embedder(settings), parents, children, vectors)


def build_prompt(question: str, sources: list[Parent]) -> str:
    blocks = [f"[S{n}] {p.path} ({p.pages})\n{p.text}" for n, p in enumerate(sources, 1)]
    return "Sources:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQuestion: {question}"


def answer(question: str, retriever: Retriever, gemini: Gemini, settings: Settings, rerank: bool | None = None) -> Answer:
    result = retriever.search(question, rerank=rerank)
    if not result.parents:
        return Answer(question, REFUSAL, result, [])
    prompt = build_prompt(question, result.parents)
    t = time.perf_counter()
    text = gemini.generate(
        prompt,
        model=settings.llm_model,
        system=SYSTEM_PROMPT,
    )
    return Answer(question, text, result, result.parents, prompt=prompt, generation_ms=elapsed_ms(t))
