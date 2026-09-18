"""
ScopedHybridRetriever — Sprint 3 extended retriever.
Builds Typesense filter_by strings from a SearchScope object,
enabling all SRCH-01 through SRCH-10 requirements.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.config import get_settings
from app.logging_config import get_logger
from app.retrieval.typesense_client import get_typesense_client, hybrid_search

log = get_logger(__name__)


@dataclass
class SearchScope:
    """
    Defines exactly which documents the retriever searches.
    All fields are optional — omitting them widens the scope.
    """
    # Who is searching
    student_id: Optional[str] = None

    # Module / course scoping
    module_id: Optional[str]    = None
    course_code: Optional[str]  = None   # SRCH-03: /csc109 shortcut

    # Semester scoping
    semester_id: Optional[str]  = None
    current_semester_only: bool = True   # SRCH-01 default

    # Week scoping
    week_number: Optional[int]  = None   # SRCH-04: /csc109/week3 shortcut
    week_id: Optional[str]      = None

    # Visibility
    include_class: bool    = True    # class materials
    include_personal: bool = True    # student's own notes
    personal_only: bool    = False   # SRCH-09: explicit personal-only mode

    # Version
    latest_only: bool = True   # DOC-04: only is_latest=true by default

    # Multiple module IDs (for "all my modules" search)
    module_ids: list[str] = field(default_factory=list)

    def build_filter(self) -> str:
        """Build a Typesense filter_by string from the scope."""
        filters: list[str] = []

        # ── Version ─────────────────────────────────────────────────────
        if self.latest_only:
            filters.append("is_latest:=true")

        # ── Semester ────────────────────────────────────────────────────
        if self.current_semester_only and not self.semester_id:
            filters.append("is_current_semester:=true")
        elif self.semester_id:
            filters.append(f"semester_id:={self.semester_id}")

        # ── Course code / module ────────────────────────────────────────
        if self.course_code:
            filters.append(f"course_code:={self.course_code.upper()}")
        elif self.module_id:
            filters.append(f"module_id:={self.module_id}")
        elif self.module_ids:
            ids_str = ",".join(self.module_ids)
            filters.append(f"module_id:[{ids_str}]")

        # ── Week ────────────────────────────────────────────────────────
        if self.week_number is not None:
            filters.append(f"week_number:={self.week_number}")
        elif self.week_id:
            filters.append(f"week_id:={self.week_id}")

        # ── Visibility / access control ─────────────────────────────────
        # SRCH-08: Personal notes are architecturally isolated
        if self.personal_only and self.student_id:
            filters.append(f"visibility:=personal && owner_id:={self.student_id}")
        elif self.include_class and self.include_personal and self.student_id:
            # Both: class materials OR this student's personal notes
            filters.append(
                f"(visibility:=class || "
                f"(visibility:=personal && owner_id:={self.student_id}))"
            )
        elif self.include_class:
            filters.append("visibility:=class")
        elif self.include_personal and self.student_id:
            filters.append(f"visibility:=personal && owner_id:={self.student_id}")

        return " && ".join(filters) if filters else ""

    @classmethod
    def from_shortcut(cls, shortcut: str, student_id: str) -> "SearchScope":
        """
        SRCH-03/04: Parse /csc109 or /csc109/week3 shortcuts.
        Returns a SearchScope with course_code (and optionally week_number) set.
        """
        parts = shortcut.lstrip("/").split("/")
        scope = cls(student_id=student_id)

        if len(parts) >= 1:
            scope.course_code = parts[0].upper()
        if len(parts) >= 2:
            week_part = parts[1].lower().replace("week", "").strip()
            try:
                scope.week_number = int(week_part)
            except ValueError:
                pass

        return scope


class ScopedHybridRetriever(BaseRetriever):
    """
    LlamaIndex BaseRetriever backed by Typesense hybrid search.
    Accepts a SearchScope object for full filter control.
    """

    def __init__(
        self,
        embed_model,
        scope: SearchScope | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self._embed_model     = embed_model
        self._client          = get_typesense_client()
        settings              = get_settings()
        self._top_k           = top_k or settings.top_k_retrieval
        self._score_threshold = score_threshold if score_threshold is not None else 0.3
        self._scope           = scope or SearchScope()
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_text = query_bundle.query_str

        # Check for /course_code/week shortcut prefix
        if query_text.startswith("/") and " " in query_text:
            parts      = query_text.split(" ", 1)
            shortcut   = parts[0]
            query_text = parts[1]
            self._scope = SearchScope.from_shortcut(
                shortcut, self._scope.student_id or ""
            )
            log.info("scope_shortcut_parsed", shortcut=shortcut, scope=str(self._scope))

        embedding  = self._embed_model.get_text_embedding(query_text)
        filter_by  = self._scope.build_filter()

        log.info("retrieval_scope", filter=filter_by, query=query_text[:80])

        hits = hybrid_search(
            client=self._client,
            query=query_text,
            embedding=embedding,
            top_k=self._top_k,
            filter_by=filter_by or None,
        )

        nodes: list[NodeWithScore] = []
        for hit in hits:
            score = float(hit.get("score", 0.0))
            if score < self._score_threshold:
                continue

            node = TextNode(
                text=hit["content"],
                id_=hit["id"],
                metadata={
                    "document_id":  hit.get("document_id", ""),
                    "filename":     hit.get("filename", ""),
                    "chunk_index":  hit.get("chunk_index", 0),
                    "token_count":  hit.get("token_count", 0),
                    "module_id":    hit.get("module_id", ""),
                    "course_code":  hit.get("course_code", ""),
                    "week_number":  hit.get("week_number", 0),
                    "visibility":   hit.get("visibility", ""),
                    "doc_version":  hit.get("doc_version", 1),
                },
            )
            nodes.append(NodeWithScore(node=node, score=score))

        log.info(
            "retrieval_complete",
            query=query_text[:80],
            hits_total=len(hits),
            hits_above_threshold=len(nodes),
            filter=filter_by,
        )
        return nodes
