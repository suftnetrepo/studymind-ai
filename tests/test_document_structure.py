"""Chapter/section detection at upload time, chunk mapping, and the retrieval filter that scopes generation to chosen sections."""
import io

import pytest

from app.db.schemas import FlashcardGenerateRequest, QuizGenerateRequest, SummariseRequest
from app.ingestion.structure import extract_with_structure, map_chunks, Section
from app.retrieval.retriever import SearchScope

MD = """# Biology Booklet

Intro text.

## Chapter 1: Cells

Cells are the basic unit of life.

```
# not a heading
```

## Chapter 2: Genetics

Genes carry information.

### 2.1 Inheritance

Mendel studied peas.
"""


def titles(sections):
    return [(s.level, s.title) for s in sections]


def test_markdown_headings_and_fenced_code_ignored():
    text, sections = extract_with_structure("book.md", MD.encode())
    assert text == MD
    assert titles(sections) == [(1, "Biology Booklet"), (2, "Chapter 1: Cells"), (2, "Chapter 2: Genetics"), (3, "2.1 Inheritance")]
    assert all(text[s.start:].startswith(("#")) for s in sections)


def test_plain_text_keyword_and_numbered_headings():
    body = "Course notes\n\nChapter 1: Forces\nF = ma applies.\n\nChapter 2 - Energy\nWork is force times distance.\n\n2.1 Kinetic energy\nHalf m v squared.\n"
    _, sections = extract_with_structure("notes.txt", body.encode())
    assert titles(sections) == [(1, "Chapter 1: Forces"), (1, "Chapter 2 - Energy"), (2, "2.1 Kinetic energy")]


def test_no_structure_is_not_invented():
    for body in ["Just a paragraph of text.\nAnother line about nothing in particular.\n", "Chapter 1: Only one heading\ntext\n", "name,age\nann,3\n"]:
        assert extract_with_structure("a.txt", body.encode())[1] == []
    assert extract_with_structure("data.csv", b"Chapter 1,x\nChapter 2,y\n")[1] == []


def test_sentences_and_long_numbered_lists_are_not_headings():
    body = "1. Mix the reagents carefully and then wait.\n2 Mole of gas occupies 22.4 litres at stp.\nChapter 3 is where we continue the story of the long journey through the whole course material today ok\n"
    assert extract_with_structure("x.txt", body.encode())[1] == []
    many = "\n".join(f"{i} Item number {i}" for i in range(1, 90))
    assert extract_with_structure("list.txt", many.encode())[1] == []


def test_docx_heading_styles():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Chemistry Handbook", 1)
    d.add_paragraph("Welcome.")
    d.add_heading("Atoms", 2)
    d.add_paragraph("Atoms have nuclei.")
    d.add_heading("Bonding", 2)
    d.add_paragraph("Ionic and covalent.")
    buf = io.BytesIO(); d.save(buf)
    text, sections = extract_with_structure("c.docx", buf.getvalue())
    assert titles(sections) == [(1, "Chemistry Handbook"), (2, "Atoms"), (2, "Bonding")]
    assert text[sections[1].start:].startswith("Atoms")


def test_pdf_bookmarks_with_pages():
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    for _ in range(3):
        w.add_blank_page(width=200, height=200)
    c1 = w.add_outline_item("Part One", 0)
    w.add_outline_item("Details", 1, parent=c1)
    w.add_outline_item("Part Two", 2)
    buf = io.BytesIO(); w.write(buf)
    _, sections = extract_with_structure("p.pdf", buf.getvalue())
    assert [(s.level, s.title, s.page) for s in sections] == [(1, "Part One", 1), (2, "Details", 2), (1, "Part Two", 3)]


def test_unsupported_type_still_raises():
    with pytest.raises(ValueError):
        extract_with_structure("a.exe", b"x")


def test_map_chunks_overlap_and_drop_unreachable():
    sections = [Section("A", 1, 0), Section("B", 1, 100), Section("C", 1, 105), Section("D", 1, 300)]
    spans = [(0, 90), (80, 200), (190, 290), (280, 400)]
    out = map_chunks(sections, spans)
    assert [(s["i"], s["title"], s["chunk_start"], s["chunk_end"]) for s in out] == [(0, "A", 0, 1), (1, "B", 1, 1), (2, "C", 1, 3), (3, "D", 3, 3)]
    assert [s["i"] for s in out] == list(range(len(out)))
    assert map_chunks(sections, [(None, None)]) == []


def test_a_section_includes_its_subsections():
    sections = [Section("Handbook", 1, 0), Section("Chapter 1", 2, 100), Section("1.1 Detail", 3, 150), Section("Chapter 2", 2, 300)]
    spans = [(0, 120), (120, 260), (260, 420), (420, 500)]
    got = {s["title"]: (s["chunk_start"], s["chunk_end"]) for s in map_chunks(sections, spans)}
    assert got["Handbook"] == (0, 3)            # a lone title covers the whole document
    assert got["Chapter 1"] == (0, 2)           # up to the next chapter, with its sub-section
    assert got["1.1 Detail"] == (1, 2)
    assert got["Chapter 2"] == (2, 3)


def test_scope_filter_only_narrows():
    base = SearchScope(student_id="s1", module_id="m1", current_semester_only=False)
    scoped = SearchScope(student_id="s1", module_id="m1", current_semester_only=False, chunk_ranges=[("d1", 3, 9), ("d2", 0, 4)])
    f = scoped.build_filter()
    assert f.startswith(base.build_filter() + " && ")                       # every access/module filter is still there, unchanged
    assert "(document_id:=d1 && chunk_index:[3..9]) || (document_id:=d2 && chunk_index:[0..4])" in f
    assert SearchScope(chunk_ranges=[]).build_filter() == SearchScope().build_filter()


def test_new_request_fields_are_optional_and_bounded():
    for model, base in [(QuizGenerateRequest, {}), (FlashcardGenerateRequest, {}), (SummariseRequest, {})]:
        r = model(module_id="00000000-0000-4000-8000-000000000001", **base)
        assert r.sections is None and r.balanced is False
        with pytest.raises(Exception):
            model(module_id="00000000-0000-4000-8000-000000000001", sections=["x:1"] * 13)


def test_backfill_detects_headings_in_stored_chunks():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("backfill_sections", pathlib.Path(__file__).parent.parent / "scripts" / "backfill_sections.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    chunks = [(0, "# Handbook\n\nintro"), (1, "## Chapter 1: Cells\n\ntext"), (2, "more text of chapter one"), (3, "## Chapter 2: Genetics\n\ntext"), (4, "tail")]
    got = mod.detect(chunks)
    assert [(s["title"], s["chunk_start"], s["chunk_end"]) for s in got] == [("Handbook", 0, 4), ("Chapter 1: Cells", 1, 3), ("Chapter 2: Genetics", 3, 4)]
    assert mod.detect([(0, "just text"), (1, "more text")]) == []
