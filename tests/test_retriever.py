"""
Unit tests — RAG pipeline citation extraction (no network needed).
"""
import pytest
from app.agents.rag_pipeline import _format_history, _extract_citations
from llama_index.core.schema import NodeWithScore, TextNode


def _make_node(doc_id, filename, chunk_idx, text, score):
    node = TextNode(
        text=text,
        id_=f"{doc_id}__{chunk_idx}",
        metadata={"document_id": doc_id, "filename": filename, "chunk_index": chunk_idx},
    )
    return NodeWithScore(node=node, score=score)


class TestFormatHistory:
    def test_empty_history(self):
        assert _format_history([]) == "(none)"

    def test_basic_history(self):
        h = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        out = _format_history(h)
        assert "User: hi" in out
        assert "Assistant: hello" in out

    def test_truncates_to_10_turns(self):
        h = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        out = _format_history(h)
        lines = [l for l in out.split("\n") if l.strip()]
        assert len(lines) == 10


class TestExtractCitations:
    def test_empty_nodes(self):
        assert _extract_citations([]) == []

    def test_basic_citation(self):
        nodes = [_make_node("doc1", "report.pdf", 0, "Some text here.", 0.9)]
        cites = _extract_citations(nodes)
        assert len(cites) == 1
        assert cites[0].filename == "report.pdf"
        assert cites[0].relevance_score == 0.9

    def test_deduplication(self):
        """Same doc_id + chunk_index should only appear once."""
        nodes = [
            _make_node("doc1", "f.pdf", 0, "Text A", 0.9),
            _make_node("doc1", "f.pdf", 0, "Text A", 0.85),
        ]
        cites = _extract_citations(nodes)
        assert len(cites) == 1

    def test_sorted_by_score_descending(self):
        nodes = [
            _make_node("doc1", "a.pdf", 0, "low",  0.6),
            _make_node("doc2", "b.pdf", 1, "high", 0.95),
            _make_node("doc3", "c.pdf", 2, "mid",  0.75),
        ]
        cites = _extract_citations(nodes)
        scores = [c.relevance_score for c in cites]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_filters_low_scores(self):
        nodes = [
            _make_node("doc1", "a.pdf", 0, "good",     0.8),
            _make_node("doc2", "b.pdf", 1, "too low",  0.3),
        ]
        cites = _extract_citations(nodes, threshold=0.6)
        assert len(cites) == 1
        assert cites[0].filename == "a.pdf"
