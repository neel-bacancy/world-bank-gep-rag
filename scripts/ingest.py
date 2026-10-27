"""Build the index: parse the PDF by structure, chunk hierarchically, embed the children, save to disk.

    python -m scripts.ingest
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from app.chunker import build_chunks, to_dict
from app.config import get_settings
from app.embedder import Embedder
from app.parser import parse_pdf


def main() -> None:
    s = get_settings()
    segments = parse_pdf(str(s.pdf_path))
    parents, children = build_chunks(segments, s)

    print(f"segments (paragraphs): {len(segments)}")
    print(f"parents (section context): {len(parents)}  median {int(np.median([len(p.text) for p in parents]))} chars")
    print(f"children (searchable):    {len(children)}  median {int(np.median([len(c.text) for c in children]))} chars")
    by_chapter = Counter(p.chapter for p in parents)
    for chapter, n in by_chapter.items():
        print(f"  {n:4d}  {chapter}")

    vectors = Embedder(s).embed([c.embed_text for c in children], task="RETRIEVAL_DOCUMENT")

    s.index_dir.mkdir(parents=True, exist_ok=True)
    payload = {"parents": [to_dict(p) for p in parents], "children": [to_dict(c) for c in children]}
    (s.index_dir / "index.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    np.save(s.index_dir / "vectors.npy", vectors)
    print(f"saved {len(children)} vectors ({vectors.shape[1]} dims) to {s.index_dir}")


if __name__ == "__main__":
    main()
