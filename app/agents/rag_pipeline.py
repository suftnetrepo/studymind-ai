"""
RAG pipeline — Sprint 4: fully scope-aware.

Flow:
  1. Build SearchScope from ChatRequest (module_id, scope_mode, shortcuts).
  2. ScopedHybridRetriever fetches top-k chunks with Typesense filter_by.
  3. Context assembled with filenames, chunk indices, week numbers.
  4. GPT-4o generates grounded answer with inline citations.
  5. Citations extracted, deduplicated, sorted by relevance score.
  6. Scope indicator built for SRCH-05 transparency.
"""
from __future__ import annotations

import time
from typing import Any

from llama_index.core import PromptTemplate
from llama_index.core.schema import QueryBundle

from app.agents.llm_factory import get_embed_model, get_llm
from app.config import get_settings
from app.db.schemas import ScopeIndicator, SourceCitation
from app.logging_config import get_logger
from app.retrieval.retriever import SearchScope, ScopedHybridRetriever

log = get_logger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Revvo, an intelligent study assistant grounded in the student's
own course materials.

Guidelines:
- Answer accurately using ONLY the provided context. Never fabricate.
- Do NOT add inline citations like [filename, chunk N] in your answer text.
  Citations are handled separately and shown below your answer automatically.
- If the context does not contain enough information, say so briefly but still answer from what you have.
  Only say you cannot answer if the context is completely empty.
- Maintain coherence across the conversation history.
- Format structured answers (lists, code, definitions) in Markdown.
- Keep answers focused and educational — you are helping a student learn.
"""


COMPLEXITY_MODIFIERS = {
    "simple": """
COMPLEXITY LEVEL: SIMPLE
- Use very simple language, short sentences, everyday analogies
- Avoid technical jargon — if you must use a technical term, immediately explain it simply
- Use the "Explain Like I'm 5" approach: relate concepts to everyday things
- Keep answers concise and friendly
""",
    "normal": """
COMPLEXITY LEVEL: NORMAL
- Use clear, accessible language suitable for a university student
- Balance technical accuracy with readability
- Use examples where helpful
""",
    "expert": """
COMPLEXITY LEVEL: EXPERT
- Use precise technical terminology without simplification
- Include implementation details, edge cases, and nuances
- Assume strong prior knowledge of the subject
- Reference related concepts and deeper theory where relevant
""",
}


def build_system_prompt(complexity: str = "normal") -> str:
    modifier = COMPLEXITY_MODIFIERS.get(complexity, COMPLEXITY_MODIFIERS["normal"])
    return f"{SYSTEM_PROMPT}\n{modifier}"

QA_PROMPT_TMPL = """\
{system_prompt}

--- Conversation history ---
{history}
----------------------------

--- Retrieved context (scope: {scope_description}) ---
{context_str}
------------------------------------------------------

Student question: {query_str}

Answer:"""

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)


# ── Helpers ────────────────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for msg in history[-10:]:   # QA-04: last 10 turns
        role    = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_citations(
    nodes_with_scores: list,
    threshold: float = 0.0,
) -> list[SourceCitation]:
    """
    Deduplicate by document_id:chunk_index, filter by threshold,
    sort by relevance descending.
    """
    citations: list[SourceCitation] = []
    seen: set[str] = set()

    for nws in nodes_with_scores:
        node  = nws.node
        score = float(nws.score or 0)
        key   = (
            f"{node.metadata.get('document_id', '')}:"
            f"{node.metadata.get('chunk_index', 0)}"
        )
        if key in seen or score < threshold:
            continue
        seen.add(key)
        citations.append(
            SourceCitation(
                document_id=node.metadata.get("document_id", ""),
                filename=node.metadata.get("filename", "unknown"),
                chunk_index=int(node.metadata.get("chunk_index", 0)),
                content_snippet=node.text[:280].replace("\n", " "),
                relevance_score=round(score, 4),
            )
        )
    return sorted(citations, key=lambda c: c.relevance_score, reverse=True)


def _build_scope_from_request(
    student_id: str,
    module_id: str | None     = None,
    course_code: str | None   = None,
    scope_mode: str           = "everything",
    include_archived: bool    = False,
    semester_id: str | None   = None,
    week_number: int | None   = None,
) -> SearchScope:
    """
    SRCH-01/02: Translate chat request params into a SearchScope.
    """
    scope = SearchScope(
        student_id=student_id,
        module_id=module_id,
        course_code=course_code,
        semester_id=semester_id,
        week_number=week_number,
        current_semester_only=not include_archived,
        latest_only=True,
    )

    if scope_mode == "class_only":
        scope.include_class    = True
        scope.include_personal = False
        scope.personal_only    = False

    elif scope_mode == "personal_only":
        scope.include_class    = False
        scope.include_personal = True
        scope.personal_only    = True

    elif scope_mode == "all_semesters":
        scope.current_semester_only = False
        scope.include_class         = True
        scope.include_personal      = True

    else:  # "everything" — default SRCH-01
        scope.include_class         = True
        scope.include_personal      = True
        scope.current_semester_only = not include_archived

    return scope


def _build_scope_indicator(
    scope: SearchScope,
    scope_mode: str,
    nodes_count: int,
    course_code: str | None     = None,
    semester_label: str | None  = None,
) -> ScopeIndicator:
    """SRCH-05: Build the human-readable scope indicator."""
    parts = []
    if scope.course_code:
        parts.append(scope.course_code)
    elif scope.module_id:
        parts.append(f"Module {scope.module_id[:8]}…")
    if scope.week_number is not None:
        parts.append(f"Week {scope.week_number}")
    if semester_label:
        parts.append(semester_label)
    if scope.current_semester_only:
        parts.append("Current semester")
    else:
        parts.append("All semesters")
    if scope_mode == "class_only":
        parts.append("Class materials only")
    elif scope_mode == "personal_only":
        parts.append("My notes only")

    return ScopeIndicator(
        description=" · ".join(parts) if parts else "All documents",
        module_id=scope.module_id,
        course_code=scope.course_code,
        semester_label=semester_label,
        week_number=scope.week_number,
        scope_mode=scope_mode,
        document_count=nodes_count,
    )


def _no_content_suggestions(scope: SearchScope) -> list[str]:
    """SRCH-06: Suggest how to widen scope when no results found."""
    suggestions = []
    if scope.week_number is not None:
        suggestions.append("Expand to all weeks in this module")
    if scope.course_code and not scope.module_id:
        suggestions.append("Search across all your enrolled modules")
    if scope.current_semester_only:
        suggestions.append("Include previous semester materials")
    if scope.personal_only:
        suggestions.append("Also search class materials")
    if scope.include_class and not scope.include_personal:
        suggestions.append("Include your personal notes")
    return suggestions or ["Try a different search query"]


# ── Main Pipeline ──────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Stateless pipeline — one singleton instance, reused across all requests.
    History and scope are passed per-call for full isolation.
    """

    def __init__(self) -> None:
        self._embed_model = get_embed_model()
        self._llm         = get_llm()

    def _make_retriever(
        self,
        scope: SearchScope,
        top_k: int,
    ) -> ScopedHybridRetriever:
        return ScopedHybridRetriever(
            embed_model=self._embed_model,
            scope=scope,
            top_k=top_k,
        )

    def query(
        self,
        question: str,
        history: list[dict] | None  = None,
        top_k: int                  = 6,
        # Sprint 4 scope params
        student_id: str             = "",
        module_id: str | None       = None,
        course_code: str | None     = None,
        scope_mode: str             = "everything",
        include_archived: bool      = False,
        semester_id: str | None     = None,
        semester_label: str | None  = None,
        week_number: int | None     = None,
        complexity: str             = "normal",
    ) -> dict:
        """
        Synchronous scoped RAG query.

        Returns:
            {
                answer, sources, latency_ms, token_count,
                scope, no_content_found, suggestions
            }
        """
        t0       = time.perf_counter()
        history  = history or []
        settings = get_settings()

        # Build scope
        scope = _build_scope_from_request(
            student_id=student_id,
            module_id=module_id,
            course_code=course_code,
            scope_mode=scope_mode,
            include_archived=include_archived,
            semester_id=semester_id,
            week_number=week_number,
        )

        retriever        = self._make_retriever(scope, top_k or settings.top_k_retrieval)
        query_bundle     = QueryBundle(query_str=question)
        nodes_with_score = retriever.retrieve(query_bundle)

        no_content   = len(nodes_with_score) == 0
        suggestions  = _no_content_suggestions(scope) if no_content else []

        # Build context string
        context_parts = []
        for i, nws in enumerate(nodes_with_score, 1):
            meta   = nws.node.metadata
            fname  = meta.get("filename", "doc")
            cidx   = meta.get("chunk_index", i)
            wk     = meta.get("week_number", 0)
            prefix = f"[{fname}, chunk {cidx}" + (f", week {wk}" if wk else "") + "]"
            context_parts.append(f"{prefix}\n{nws.node.text}")

        context_str = (
            "\n\n".join(context_parts)
            if context_parts
            else "(no relevant content found in the selected scope)"
        )

        # Scope indicator for prompt
        scope_desc = scope.course_code or (
            f"module {module_id[:8]}" if module_id else "all materials"
        )

        prompt = QA_PROMPT.format(
            system_prompt=build_system_prompt(complexity),
            history=_format_history(history),
            scope_description=scope_desc,
            context_str=context_str,
            query_str=question,
        )

        response    = self._llm.complete(prompt)
        answer_text = str(response)
        latency_ms  = int((time.perf_counter() - t0) * 1000)

        token_count = None
        try:
            if hasattr(response, "raw") and response.raw:
                raw = response.raw
                if hasattr(raw, "usage") and raw.usage:
                    token_count = raw.usage.total_tokens
                elif hasattr(raw, "get"):
                    usage = raw.get("usage", {})
                    token_count = usage.get("total_tokens")
        except Exception:
            token_count = None

        citations      = _extract_citations(nodes_with_score)
        scope_indicator = _build_scope_indicator(
            scope, scope_mode, len(nodes_with_score), course_code, semester_label
        )

        log.info(
            "rag_query_complete",
            question=question[:80],
            scope=scope_desc,
            scope_mode=scope_mode,
            chunks_retrieved=len(nodes_with_score),
            citations=len(citations),
            latency_ms=latency_ms,
            no_content=no_content,
        )

        return {
            "answer":           answer_text,
            "sources":          citations,
            "latency_ms":       latency_ms,
            "token_count":      token_count,
            "scope":            scope_indicator,
            "no_content_found": no_content,
            "suggestions":      suggestions,
        }

    def stream_query(
        self,
        question: str,
        history: list[dict] | None  = None,
        top_k: int                  = 6,
        student_id: str             = "",
        module_id: str | None       = None,
        course_code: str | None     = None,
        scope_mode: str             = "everything",
        include_archived: bool      = False,
        semester_id: str | None     = None,
        semester_label: str | None  = None,
        week_number: int | None     = None,
        complexity: str             = "normal",
    ) -> tuple[Any, list[SourceCitation], ScopeIndicator]:
        """
        Returns (stream, citations, scope_indicator).
        Caller iterates stream for tokens then sends done event with citations + scope.
        """
        history  = history or []
        settings = get_settings()

        scope = _build_scope_from_request(
            student_id=student_id,
            module_id=module_id,
            course_code=course_code,
            scope_mode=scope_mode,
            include_archived=include_archived,
            semester_id=semester_id,
            week_number=week_number,
        )

        retriever        = self._make_retriever(scope, top_k or settings.top_k_retrieval)
        query_bundle     = QueryBundle(query_str=question)
        nodes_with_score = retriever.retrieve(query_bundle)

        context_parts = []
        for i, nws in enumerate(nodes_with_score, 1):
            meta   = nws.node.metadata
            fname  = meta.get("filename", "doc")
            cidx   = meta.get("chunk_index", i)
            wk     = meta.get("week_number", 0)
            prefix = f"[{fname}, chunk {cidx}" + (f", week {wk}" if wk else "") + "]"
            context_parts.append(f"{prefix}\n{nws.node.text}")

        context_str = (
            "\n\n".join(context_parts)
            if context_parts
            else "(no relevant content found in the selected scope)"
        )

        scope_desc = scope.course_code or (
            f"module {module_id[:8]}" if module_id else "all materials"
        )

        prompt = QA_PROMPT.format(
            system_prompt=build_system_prompt(complexity),
            history=_format_history(history),
            scope_description=scope_desc,
            context_str=context_str,
            query_str=question,
        )

        stream          = self._llm.stream_complete(prompt)
        citations       = _extract_citations(nodes_with_score)
        scope_indicator = _build_scope_indicator(
            scope, scope_mode, len(nodes_with_score), course_code, semester_label
        )
        return stream, citations, scope_indicator


# Singleton
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
