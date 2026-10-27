"""Structure-aware PDF parser for the World Bank Global Economic Prospects report.

The report is a two-column layout with a clear typographic hierarchy, so structure comes from fonts
rather than from guessing at text:

  CHAPTER n (26pt)  + chapter title (30pt)   -> chapter
  Front/back matter titles (26pt)            -> chapter-level label ("Executive Summary", "References")
  Section (15.5pt+ bold)                     -> section   ("South Asia", "Global context")
  Subsection (13.5-15.5pt bold, or short bold-italic) -> subsection ("Outlook", "Downside risks")
  BOX n.n Title (10pt bold)                  -> section   (boxed articles stay separate from prose)

Reading order: the body is two columns, so blocks are read left column first, then right column, within
each band between full-width blocks (headings, figures). Running headers/footers are dropped by position.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pymupdf

PAGE_TOP = 60  # points: running header band
PAGE_BOTTOM = 45  # points: running footer band
CHAPTER_RE = re.compile(r"^CHAPTER\s+(\d+)$", re.I)
BOX_RE = re.compile(r"^BOX\s+\d+\.\d+\b", re.I)
DOT_LEADER_RE = re.compile(r"(\.\s?){5,}")
SECTION_MIN_PT = 15.5
SUBSECTION_MIN_PT = 13.5
TITLE_MIN_PT = 25.0


@dataclass
class Segment:
    """One paragraph of body text, tagged with the heading path it sits under."""

    chapter: str | None
    section: str | None
    subsection: str | None
    page: int
    text: str


@dataclass
class Line:
    text: str
    size: float
    font: str

    @property
    def bold(self) -> bool:
        return "Bold" in self.font

    @property
    def bold_italic(self) -> bool:
        return "Bold" in self.font and ("Italic" in self.font or "Obl" in self.font)


@dataclass
class Block:
    bbox: tuple[float, float, float, float]
    lines: list[Line]

    @property
    def max_size(self) -> float:
        return max(line.size for line in self.lines)

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)


def normalise(text: str) -> str:
    """Repair text-extraction artefacts before any chunking or indexing."""
    t = unicodedata.normalize("NFKC", text)  # ligatures: "ﬁ" -> "fi", "ﬂ" -> "fl"
    t = t.replace(" ", " ").replace("­", "")  # line separator, soft hyphen
    t = re.sub(r"\s+", " ", t).strip()
    # The source PDF drops the "h" of "Th" in some drop caps ("Te global outlook"). "Te" and "te"
    # never occur as standalone words in this report, so restoring them is safe.
    t = re.sub(r"\bTe\b", "The", t)
    t = re.sub(r"\bte\b", "the", t)
    return t


def _page_blocks(page: pymupdf.Page) -> tuple[list[Block], float]:
    width, height = page.rect.width, page.rect.height
    blocks: list[Block] = []
    for raw in page.get_text("dict")["blocks"]:
        if raw["type"] != 0:
            continue
        lines: list[Line] = []
        for line in raw["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            y0, y1 = line["bbox"][1], line["bbox"][3]
            if not spans or y0 < PAGE_TOP or y1 > height - PAGE_BOTTOM:
                continue
            dominant = max(spans, key=lambda s: len(s["text"]))
            lines.append(
                Line(
                    text=normalise("".join(s["text"] for s in spans)),
                    size=round(dominant["size"], 1),
                    font=dominant["font"],
                )
            )
        if lines:
            blocks.append(Block(bbox=tuple(raw["bbox"]), lines=lines))
    return blocks, width


def _reading_order(blocks: list[Block], width: float) -> list[Block]:
    """Left column then right column within each band; full-width blocks break the bands."""
    ordered: list[Block] = []
    band: list[Block] = []

    def flush() -> None:
        mid = width / 2 - 10
        left = sorted((b for b in band if b.bbox[0] < mid), key=lambda b: b.bbox[1])
        right = sorted((b for b in band if b.bbox[0] >= mid), key=lambda b: b.bbox[1])
        ordered.extend(left + right)
        band.clear()

    for block in sorted(blocks, key=lambda b: b.bbox[1]):
        if block.bbox[2] - block.bbox[0] > 0.55 * width:
            flush()
            ordered.append(block)
        else:
            band.append(block)
    flush()
    return ordered


def _is_toc(blocks: list[Block]) -> bool:
    """Contents pages use dot leaders; they would otherwise match every heading query."""
    return sum(1 for b in blocks for line in b.lines if DOT_LEADER_RE.search(line.text)) >= 5


def _join_lines(lines: list[Line]) -> str:
    """Join wrapped lines; rejoin words hyphenated across a line break ("com-" + "modity")."""
    out = ""
    for line in lines:
        if out.endswith("-") and line.text[:1].islower():
            out = out[:-1] + line.text
        else:
            out = f"{out} {line.text}".strip()
    return out


def parse_pdf(path: str) -> list[Segment]:
    doc = pymupdf.open(path)
    segments: list[Segment] = []
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    title_open = False  # a multi-block title (e.g. 30pt + 26pt lines) is still being read

    for page_no, page in enumerate(doc, start=1):
        blocks, width = _page_blocks(page)
        if _is_toc(blocks):
            continue  # contents pages would match every heading query; skip them
        title_open = False  # continuation titles only join a title that started on this page
        ordered = _reading_order(blocks, width)

        # A chapter opener is a "CHAPTER n" label plus one or more title blocks on the same page. The
        # title block can sit above the label in reading order, so pair them per page, not per block.
        titles = [b for b in ordered if b.max_size >= TITLE_MIN_PT]
        label = next((m for b in titles if (m := CHAPTER_RE.match(b.text))), None)
        if label:
            name_parts = [b.text for b in titles if not CHAPTER_RE.match(b.text)]
            chapter_name = " ".join(name_parts)
            chapter = f"Chapter {label.group(1)}: {chapter_name.capitalize() if chapter_name.isupper() else chapter_name}"
            section = subsection = None
            ordered = [b for b in ordered if b not in titles]
            title_open = False

        for block in ordered:
            text = block.text
            first = block.lines[0]

            if block.max_size >= TITLE_MIN_PT:
                if title_open and chapter and not label:
                    chapter = f"{chapter} {text}"  # subtitle block of a front-matter title
                else:
                    # Front or back matter ("Executive Summary", "References"): its own top-level label.
                    chapter = text.capitalize() if text.isupper() else text
                    section = subsection = None
                    title_open = True
                continue

            title_open = False

            if BOX_RE.match(text) and first.bold:
                section = re.sub(r"\s*\(continued\)", "", text).strip()
                subsection = None
                continue

            if first.bold and block.max_size >= SECTION_MIN_PT:
                if text != section:
                    section, subsection = text, None
                continue

            if first.bold and block.max_size >= SUBSECTION_MIN_PT:
                subsection = text
                continue

            if all(line.bold_italic for line in block.lines) and len(text) < 110 and not text.endswith("."):
                subsection = text
                continue

            body = _join_lines(block.lines)
            if section == "References":
                continue  # bibliography entries contain no findings; they only add noise to search
            if body:
                segments.append(Segment(chapter, section, subsection, page_no, body))

    return segments
