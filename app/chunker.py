"""Hierarchical (parent-child) chunking that follows the document's own structure.

- A PARENT is one heading path's text (chapter > section > subsection), packed up to ~900 tokens. The
  answer model reads parents, so it gets the whole section, not a fragment.
- A CHILD is a ~280-token sentence window inside one parent. Children are what get searched, because
  small units match a specific fact (one region's forecast) far more precisely than a whole chapter.
- Every child stores its parent_id. A child hit is resolved up to its parent, so search stays precise
  while the answer keeps its context.
- The text that is embedded and indexed for BM25 starts with the heading path, so "South Asia" in a
  question matches chunks under that heading even when the body never repeats the region name. The
  stored text stays unchanged for display and citations.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.config import Settings
from app.parser import Segment

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(\[])')


@dataclass
class Parent:
    id: str
    chapter: str | None
    section: str | None
    subsection: str | None
    region: str | None
    page_start: int
    page_end: int
    text: str

    @property
    def path(self) -> str:
        return heading_path(self.chapter, self.section, self.subsection)

    @property
    def pages(self) -> str:
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"


@dataclass
class Child:
    id: str
    parent_id: str
    text: str  # what the user sees and what the answer cites
    embed_text: str  # heading path + text: what is embedded and matched with BM25
    chapter: str | None
    section: str | None
    subsection: str | None
    region: str | None
    page: int

    @property
    def path(self) -> str:
        return heading_path(self.chapter, self.section, self.subsection)


def heading_path(chapter: str | None, section: str | None, subsection: str | None) -> str:
    return " > ".join(part for part in (chapter, section, subsection) if part)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def _pack_units(units: list[str], max_chars: int, overlap: int = 0) -> list[str]:
    """Greedy packing of sentences (or paragraphs) into windows, with optional sentence overlap."""
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) + 1 > max_chars:
            windows.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            size = sum(len(u) + 1 for u in current)
        current.append(unit)
        size += len(unit) + 1
    if current:
        windows.append(" ".join(current))
    return windows


def build_chunks(segments: list[Segment], settings: Settings) -> tuple[list[Parent], list[Child]]:
    # 1. Group consecutive paragraphs that share the same heading path.
    groups: list[tuple[tuple, list[Segment]]] = []
    for seg in segments:
        key = (seg.chapter, seg.section, seg.subsection)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(seg)
        else:
            groups.append((key, [seg]))

    parents: list[Parent] = []
    children: list[Child] = []

    for (chapter, section, subsection), segs in groups:
        region = section if chapter and "Regional" in chapter else None

        # 2. Pack paragraphs into parents of at most parent_max_chars.
        packed: list[tuple[int, int, str]] = []
        for seg in segs:
            paragraph = seg.text
            if packed and len(packed[-1][2]) + len(paragraph) + 2 <= settings.parent_max_chars:
                start, _, text = packed[-1]
                packed[-1] = (start, seg.page, f"{text}\n\n{paragraph}")
            else:
                packed.append((seg.page, seg.page, paragraph))

        for start, end, parent_text in packed:
            parent = Parent(
                id=f"P{len(parents):04d}",
                chapter=chapter,
                section=section,
                subsection=subsection,
                region=region,
                page_start=start,
                page_end=end,
                text=parent_text,
            )
            parents.append(parent)

            # 3. Sentence windows inside the parent become the searchable children.
            for window in _pack_units(_sentences(parent_text), settings.child_max_chars, settings.child_overlap_sentences):
                cid = f"C{len(children):05d}"
                children.append(
                    Child(
                        id=cid,
                        parent_id=parent.id,
                        text=window,
                        embed_text=f"{parent.path}\n{window}",
                        chapter=chapter,
                        section=section,
                        subsection=subsection,
                        region=region,
                        page=start,
                    )
                )

    return parents, children


def to_dict(obj) -> dict:
    return asdict(obj)
