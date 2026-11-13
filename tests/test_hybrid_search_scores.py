"""Hybrid search must give every hit an honest similarity so the similarity threshold cannot be bypassed."""
import io
import json
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.retrieval import typesense_client as ts


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(hits, sent):
    def urlopen(req, timeout=None):
        sent.append(json.loads(req.data))
        return _Resp(json.dumps({"results": [{"hits": hits}]}).encode())
    return urlopen


def hit(cid, distance=None):
    h = {"document": {"id": cid, "content": f"chunk {cid}"}}
    if distance is not None:
        h["vector_distance"] = distance
    return h


def run(monkeypatch, hits, top_k=6):
    sent = []
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(hits, sent))
    out = ts.hybrid_search(SimpleNamespace(), "q", [0.1, 0.2], top_k=top_k, filter_by="module_id:=m")
    return out, sent


def test_score_is_cosine_similarity_for_vector_hits(monkeypatch):
    out, _ = run(monkeypatch, [hit("a", 0.30), hit("b", 0.75)])
    assert [round(h["score"], 3) for h in out] == [0.70, 0.25]


def test_keyword_only_hit_does_not_get_an_invented_high_score(monkeypatch):
    out, _ = run(monkeypatch, [hit("kw-only"), hit("vec", 0.5)])
    by = {h["id"]: h["score"] for h in out}
    assert by["kw-only"] == 0.0                     # unknown similarity, NOT 0.75
    assert by["vec"] == 0.5


@pytest.mark.parametrize("threshold", [0.10, 0.15, 0.20, 0.25, 0.30])
def test_threshold_applies_to_keyword_only_hits(monkeypatch, threshold):
    out, _ = run(monkeypatch, [hit("kw-only"), hit("weak", 0.9), hit("good", 0.4)])
    kept = [h["id"] for h in out if h["score"] >= threshold]
    assert "kw-only" not in kept and "weak" not in kept and "good" in kept


def test_vector_candidate_depth_exceeds_the_page_size(monkeypatch):
    _, sent = run(monkeypatch, [hit("a", 0.3)], top_k=6)
    s = sent[0]["searches"][0]
    assert s["per_page"] == 6
    assert f"k:{get_settings().retrieval_vector_k})" in s["vector_query"] and get_settings().retrieval_vector_k >= 50


def test_vector_depth_never_drops_below_top_k(monkeypatch):
    monkeypatch.setattr(get_settings(), "retrieval_vector_k", 5)
    _, sent = run(monkeypatch, [hit("a", 0.3)], top_k=20)
    assert "k:20)" in sent[0]["searches"][0]["vector_query"]


def test_embedding_failure_still_falls_back_to_keyword_search_by_design(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("down")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    client = SimpleNamespace(collections={ts.COLLECTION_NAME: SimpleNamespace(documents=SimpleNamespace(
        search=lambda params: {"hits": [{"document": {"id": "k", "content": "x"}}]}))})
    out = ts.hybrid_search(client, "q", [0.1], top_k=3)
    assert [h["id"] for h in out] == ["k"] and out[0]["score"] == 0.75      # unchanged documented fallback
