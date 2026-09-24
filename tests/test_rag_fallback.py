"""RAGPipeline.query fallback_top_k — broad questions still get scoped context. No live services."""
from llama_index.core.schema import NodeWithScore, TextNode

from app.agents.rag_pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, nodes):
        self.nodes = nodes

    def retrieve(self, _query):
        return self.nodes


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return "answer"


def node(text, score):
    return NodeWithScore(
        node=TextNode(text=text, metadata={"filename": "cv.pdf", "document_id": "d1", "chunk_index": 0}),
        score=score,
    )


def make_pipeline(primary, fallback):
    """primary: nodes above threshold; fallback: nodes returned with threshold 0."""
    p = RAGPipeline.__new__(RAGPipeline)  # skip model loading
    p._llm   = FakeLLM()
    p.calls  = []

    def make_retriever(scope, top_k, score_threshold=None):
        p.calls.append((top_k, score_threshold))
        return FakeRetriever(fallback if score_threshold == 0.0 else primary)

    p._make_retriever = make_retriever
    return p


def test_fallback_used_when_nothing_clears_threshold():
    p = make_pipeline(primary=[], fallback=[node("Python developer with RAG experience", 0.12)])
    r = p.query("What is this document about?", module_id="m1", fallback_top_k=4)
    assert r["no_content_found"] is False
    assert [s.filename for s in r["sources"]] == ["cv.pdf"]
    assert p.calls[-1] == (4, 0.0)
    assert "Python developer with RAG experience" in p._llm.prompts[0]


def test_no_fallback_by_default():
    p = make_pipeline(primary=[], fallback=[node("x", 0.1)])
    r = p.query("What is this document about?", module_id="m1")
    assert r["no_content_found"] is True
    assert len(p.calls) == 1


def test_fallback_not_used_when_primary_has_results():
    p = make_pipeline(primary=[node("relevant", 0.8)], fallback=[node("other", 0.1)])
    r = p.query("Which languages?", module_id="m1", fallback_top_k=4)
    assert len(p.calls) == 1 and r["no_content_found"] is False


def test_fallback_with_empty_module_still_reports_no_content():
    p = make_pipeline(primary=[], fallback=[])
    r = p.query("anything", module_id="m1", fallback_top_k=4)
    assert r["no_content_found"] is True
