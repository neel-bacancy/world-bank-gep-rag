"""Ask one question from the terminal and see the route, the sections used, and the answer.

    python -m scripts.ask "What is the growth outlook for South Asia in this report?"
    python -m scripts.ask --no-rerank "..."
"""

from __future__ import annotations

import argparse

from app.answer import answer, load_retriever
from app.config import get_settings
from app.llm import Gemini


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    gemini = Gemini(settings)
    retriever = load_retriever(settings, gemini)
    result = answer(args.question, retriever, gemini, settings, rerank=not args.no_rerank)

    print(f"Route:   {result.search.route.describe()}")
    print(f"Search query: {result.search.route.query}")
    print("Sections used:")
    for n, p in enumerate(result.sources, 1):
        print(f"  [S{n}] {p.path} ({p.pages})")
    print()
    print(result.text)


if __name__ == "__main__":
    main()
