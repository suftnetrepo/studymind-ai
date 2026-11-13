"""
Document structure (chapters / sections), extracted once at upload time from the file itself.

Deterministic and free of AI calls: PDF bookmarks, Word heading styles, Markdown headings, and for plain text
(or PDFs/Word files without any of those) conservative "Chapter 3" / "2.1 Cell division" heading detection.
Offsets are character offsets into the extracted text, so the ingestor can map every chunk to the sections it overlaps.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

MAX_SECTIONS = 200
MAX_TITLE = 120
MIN_SECTIONS = 2          # one heading is not a structure

_KEYWORD = re.compile(
    r"^(?P<kw>chapter|unit|part|module|week|lecture|section|topic)\s+(?P<num>\d{1,3}|[ivxlc]{1,6})\b[\s:.\-–—)]*(?P<rest>.*)$",
    re.IGNORECASE,
)
_NUMBERED = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,2})[.)]?\s+(?P<title>[A-Z][^\n]{2,90})$")
_LEVEL_ONE = {"chapter", "unit", "part", "module", "week"}
_MD_HEADING = re.compile(r"^(#{1,4})[ \t]+(.+?)[ \t]*#*[ \t]*$")


@dataclass
class Section:
    title: str
    level: int            # 1 = chapter, 2 = section, 3 = sub-section
    start: int            # character offset in the extracted text
    page: int | None = None


def _clean(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip(" \t:.-–—")[:MAX_TITLE]


def _finish(sections: list[Section]) -> list[Section]:
    """Sorted by position, one per start offset, no repeated neighbours, capped. Fewer than MIN_SECTIONS means 'no structure'."""
    out: list[Section] = []
    for s in sorted(sections, key=lambda x: x.start):
        s.title = _clean(s.title)
        if len(s.title) < 2 or not re.search(r"[A-Za-z0-9]", s.title):
            continue
        if out and (out[-1].start == s.start or out[-1].title.lower() == s.title.lower()):
            continue
        out.append(s)
        if len(out) >= MAX_SECTIONS:
            break
    return out if len(out) >= MIN_SECTIONS else []


def _lines(text: str):
    pos = 0
    for line in text.split("\n"):
        yield pos, line
        pos += len(line) + 1


def headings_from_markdown(text: str) -> list[Section]:
    out, fenced = [], False
    for pos, line in _lines(text):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = _MD_HEADING.match(line)
        if m:
            out.append(Section(m.group(2), min(len(m.group(1)), 3), pos))
    return out


def headings_from_text(text: str) -> list[Section]:
    """Plain-text heading detection. Deliberately conservative: a missed heading is harmless, a wrong one is confusing."""
    keyword, numbered = [], []
    for pos, raw in _lines(text):
        line = raw.strip()
        if not line or len(line) > 100:
            continue
        m = _KEYWORD.match(line)
        if m and (not m.group("rest") or len(m.group("rest").split()) <= 12):
            level = 1 if m.group("kw").lower() in _LEVEL_ONE else 2
            keyword.append(Section(line, level, pos + (len(raw) - len(raw.lstrip()))))
            continue
        n = _NUMBERED.match(line)
        if n and not line.endswith((".", ",", ";", ":")) and len(n.group("title").split()) <= 10:
            numbered.append(Section(line, min(n.group("num").count(".") + 1, 3), pos + (len(raw) - len(raw.lstrip()))))
    # Hundreds of "1 Something" lines are a numbered list or a table, not headings.
    return keyword + (numbered if len(numbered) <= 60 else [])


def _pdf(content: bytes) -> tuple[str, list[Section]]:
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages = [p.extract_text() or "" for p in reader.pages]
    text = "\n\n".join(pages)
    starts, pos = [], 0
    for p in pages:
        starts.append(pos)
        pos += len(p) + 2
    found: list[Section] = []
    try:
        def walk(items, depth):
            for item in items:
                if isinstance(item, list):
                    walk(item, depth + 1)
                    continue
                title = getattr(item, "title", None)
                if not title:
                    continue
                page = reader.get_destination_page_number(item)
                if page is None or page < 0 or page >= len(pages):
                    continue
                at = pages[page].find(str(title).strip())
                found.append(Section(str(title), min(depth, 3), starts[page] + (at if at >= 0 else 0), page + 1))
        walk(reader.outline, 1)
    except Exception:
        found = []
    sections = _finish(found)
    return text, sections or _finish(headings_from_text(text))


def _docx(content: bytes) -> tuple[str, list[Section]]:
    import docx
    document = docx.Document(io.BytesIO(content))
    parts, found, pos = [], [], 0
    for p in document.paragraphs:
        if not p.text:
            continue
        style = (p.style.name if p.style is not None else "") or ""
        m = re.match(r"^(?:Heading|Title)\s*(\d)?$", style, re.IGNORECASE)
        if m:
            found.append(Section(p.text, min(int(m.group(1) or 1), 3), pos))
        parts.append(p.text)
        pos += len(p.text) + 2
    text = "\n\n".join(parts)
    sections = _finish(found)
    return text, sections or _finish(headings_from_text(text))


def extract_with_structure(filename: str, content: bytes) -> tuple[str, list[Section]]:
    """The extracted text (identical to what indexing has always used) and its detected sections ([] if none)."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            return _pdf(content)
        except ImportError:
            text = content.decode("utf-8", errors="replace")
            return text, _finish(headings_from_text(text))
    if ext == ".docx":
        try:
            return _docx(content)
        except ImportError:
            text = content.decode("utf-8", errors="replace")
            return text, _finish(headings_from_text(text))
    if ext in (".txt", ".md", ".csv"):
        text = content.decode("utf-8", errors="replace")
        if ext == ".csv":
            return text, []
        found = headings_from_markdown(text) if ext == ".md" else []
        return text, _finish(found) or _finish(headings_from_text(text))
    raise ValueError(f"Unsupported file type: {ext}")


def map_chunks(sections: list[Section], spans: list[tuple[int | None, int | None]]) -> list[dict]:
    """
    For each section: the first and last chunk that overlap it. `spans` is (start, end) per chunk, in order.
    Sections that no chunk overlaps are dropped; the result is numbered from 0 (that number is the section's id within the document).
    Each section covers its sub-sections too.
    """
    out: list[dict] = []
    for i, s in enumerate(sections):
        # A section runs until the next heading of the same or a higher level, so a chapter includes its sub-sections.
        end = next((t.start for t in sections[i + 1:] if t.level <= s.level), float("inf"))
        hit = [idx for idx, (a, b) in enumerate(spans) if a is not None and b is not None and a < end and b > s.start]
        if not hit:
            continue
        out.append({"i": len(out), "title": s.title, "level": s.level, "page": s.page, "chunk_start": min(hit), "chunk_end": max(hit)})
    return out
