"""
Add chapters/sections to documents that were indexed before structure detection existed.

The original files are not re-read: headings are detected in the text of each stored chunk (Markdown headings, "Chapter 3",
"2.1 Cell division"), so PDF bookmarks and Word heading styles are NOT available here - re-upload a file for those.
Only touches documents.doc_metadata["sections"]; chunks and the search index are untouched. Dry run unless --apply.

    PYTHONPATH=. python scripts/backfill_sections.py            # report what would change
    PYTHONPATH=. python scripts/backfill_sections.py --apply
"""
import sys

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Document, DocumentChunk
from app.ingestion.structure import headings_from_markdown, headings_from_text, MAX_SECTIONS, MAX_TITLE, MIN_SECTIONS


def detect(chunks: list[tuple[int, str]]) -> list[dict]:
    found = []   # (chunk_index, level, title)
    for idx, content in chunks:
        heads = headings_from_markdown(content) or headings_from_text(content)
        for h in heads:
            title = " ".join(h.title.split()).strip(" \t:.-–—")[:MAX_TITLE]
            if len(title) >= 2 and not (found and found[-1][2].lower() == title.lower()):
                found.append((idx, h.level, title))
    if len(found) < MIN_SECTIONS:
        return []
    found = found[:MAX_SECTIONS]
    out = []
    last = chunks[-1][0]
    for i, (idx, level, title) in enumerate(found):
        # up to (and including) the chunk where the next heading of the same or a higher level starts, like detection at upload
        nxt = next((c for c, lv, _ in found[i + 1:] if lv <= level), None)
        out.append({"i": i, "title": title, "level": level, "page": None, "chunk_start": idx, "chunk_end": nxt if nxt is not None else last})
    return out


def main(apply: bool) -> None:
    engine = create_engine(get_settings().postgres_dsn_sync)
    changed = 0
    with Session(engine) as session:
        for doc in session.scalars(select(Document).where(Document.status == "indexed")):
            if (doc.doc_metadata or {}).get("sections"):
                continue
            rows = session.execute(
                select(DocumentChunk.chunk_index, DocumentChunk.content).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)
            ).all()
            if not rows:
                continue
            sections = detect([(r[0], r[1]) for r in rows])
            if not sections:
                continue
            changed += 1
            print(f"{doc.filename}: {len(sections)} sections - " + "; ".join(s["title"] for s in sections[:4]) + (" ..." if len(sections) > 4 else ""))
            if apply:
                doc.doc_metadata = {**(doc.doc_metadata or {}), "sections": sections}
        if apply:
            session.commit()
    print(f"{changed} document(s) {'updated' if apply else 'would be updated (dry run; pass --apply)'}")


if __name__ == "__main__":
    main("--apply" in sys.argv)
