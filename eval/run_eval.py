"""Evaluate retrieval and answers against hand-checked questions.

    python -m eval.run_eval               # reranker on (default from config)
    python -m eval.run_eval --no-rerank   # reranker off, to measure its effect
    python -m eval.run_eval --compare     # both, side by side

A question passes when:
  * retrieval: a top-k parent has the expected chapter (and section, when one is given), and
  * answer:    every must_contain pattern matches, no forbid pattern matches, and refusals refuse.
               Patterns are regular expressions, so "4(\.0)? percent" accepts "4 percent" and "4.0 percent".
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from app.answer import answer, load_retriever
from app.config import ROOT, get_settings
from app.llm import Gemini

QUESTIONS = ROOT / "eval" / "sample_questions.json"
RESULTS = ROOT / "eval" / "results"


def retrieval_hit(question: dict, parents) -> bool:
    if not question.get("expected_chapter") and not question.get("expected_section"):
        return True  # nothing to match: scored by refusal instead
    for p in parents:
        chapter_ok = not question.get("expected_chapter") or (p.chapter or "").startswith(question["expected_chapter"])
        section_ok = not question.get("expected_section") or p.section == question["expected_section"]
        if chapter_ok and section_ok:
            return True
    return False


def answer_ok(question: dict, text: str, refused: bool) -> bool:
    if question.get("refuse"):
        return refused
    if refused:
        return False
    return all(re.search(p, text, re.I) for p in question["must_contain"]) and not any(
        re.search(p, text, re.I) for p in question["forbid"]
    )


def run(rerank: bool, retriever, gemini, settings, questions) -> dict:
    rows = []
    for q in questions:
        result = answer(q["question"], retriever, gemini, settings, rerank=rerank)
        hit = retrieval_hit(q, result.search.parents)
        ok = answer_ok(q, result.text, result.refused)
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "route": result.search.route.describe(),
                "top_parents": [f"{p.path} ({p.pages})" for p in result.sources],
                "retrieval_hit": hit,
                "answer_pass": ok,
                "passed": hit and ok,
                "answer": result.text,
            }
        )
        status = "PASS" if hit and ok else "FAIL"
        print(f"[{status}] {q['id']} retrieval={'ok' if hit else 'MISS'} answer={'ok' if ok else 'BAD'} | {q['question']}")
    total = len(rows)
    return {
        "rerank": rerank,
        "passed": sum(r["passed"] for r in rows),
        "total": total,
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in rows) / total,
        "answer_pass_rate": sum(r["answer_pass"] for r in rows) / total,
        "rows": rows,
    }


def write_report(runs: list[dict], stamp: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stamp}.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Evaluation {stamp}", ""]
    lines += ["| Reranker | Passed | Retrieval hit | Answer pass |", "|---|---|---|---|"]
    for r in runs:
        lines.append(
            f"| {'on' if r['rerank'] else 'off'} | {r['passed']}/{r['total']} | "
            f"{r['retrieval_hit_rate']:.0%} | {r['answer_pass_rate']:.0%} |"
        )
    for r in runs:
        lines += ["", f"## Reranker {'on' if r['rerank'] else 'off'}", ""]
        for row in r["rows"]:
            lines += [
                f"### {row['id']}: {row['question']}",
                f"- Result: **{'PASS' if row['passed'] else 'FAIL'}** (retrieval {'ok' if row['retrieval_hit'] else 'miss'}, answer {'ok' if row['answer_pass'] else 'bad'})",
                f"- Route: {row['route']}",
                f"- Top parents: {'; '.join(row['top_parents']) or 'none'}",
                "",
                row["answer"],
                "",
            ]
    path = RESULTS / f"{stamp}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--compare", action="store_true", help="run with and without the reranker")
    args = parser.parse_args()

    settings = get_settings()
    gemini = Gemini(settings)
    retriever = load_retriever(settings, gemini)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    modes = [True, False] if args.compare else [not args.no_rerank]
    runs = []
    for mode in modes:
        print(f"\n== reranker {'on' if mode else 'off'} ==")
        runs.append(run(mode, retriever, gemini, settings, questions))
    path = write_report(runs, datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
