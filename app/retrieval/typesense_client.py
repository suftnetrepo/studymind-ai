"""
Typesense integration — Sprint 3 extended schema.
DOC-04: Every chunk tagged with course_code, module_id, semester_id,
        week_number, visibility, owner_id, is_latest, version, is_current_semester.
"""
from __future__ import annotations

import typesense
from typesense.exceptions import ObjectNotFound
from app.config import get_settings
from app.logging_config import get_logger

log = get_logger(__name__)

COLLECTION_NAME = "document_chunks"

# DOC-04: Full extended schema with all scoping fields
COLLECTION_SCHEMA = {
    "name": COLLECTION_NAME,
    "fields": [
        # ── Core ────────────────────────────────────────────────────────
        {"name": "id",                    "type": "string"},
        {"name": "content",               "type": "string"},
        {"name": "embedding",             "type": "float[]", "num_dim": 3072, "index": True},

        # ── Document identity ────────────────────────────────────────────
        {"name": "document_id",           "type": "string", "facet": True},
        {"name": "filename",              "type": "string", "facet": True},
        {"name": "chunk_index",           "type": "int32"},
        {"name": "token_count",           "type": "int32"},

        # ── Module / course scoping ──────────────────────────────────────
        {"name": "module_id",             "type": "string", "facet": True},
        {"name": "course_code",           "type": "string", "facet": True},
        # e.g. "CSC109" — enables /csc109 scope shortcut (SRCH-03)

        # ── Semester scoping ─────────────────────────────────────────────
        {"name": "semester_id",           "type": "string", "facet": True},
        {"name": "is_current_semester",   "type": "bool",   "facet": True},

        # ── Week scoping ─────────────────────────────────────────────────
        {"name": "week_id",               "type": "string", "facet": True, "optional": True},
        {"name": "week_number",           "type": "int32",  "facet": True},

        # ── Access control ───────────────────────────────────────────────
        {"name": "visibility",            "type": "string", "facet": True},
        # class | personal
        {"name": "owner_id",              "type": "string", "facet": True},
        # student UUID for personal notes — SRCH-08 isolation

        # ── Version control ───────────────────────────────────────────────
        {"name": "is_latest",             "type": "bool",   "facet": True},
        {"name": "doc_version",           "type": "int32"},

        # ── Institution scoping ───────────────────────────────────────────
        {"name": "institution_id",        "type": "string", "facet": True, "optional": True},
        {"name": "lecturer_id",           "type": "string", "facet": True},
    ],
    "default_sorting_field": "chunk_index",
}


def get_typesense_client() -> typesense.Client:
    settings = get_settings()
    return typesense.Client({
        "nodes": [{
            "host":     settings.typesense_host,
            "port":     str(settings.typesense_port),
            "protocol": settings.typesense_protocol,
        }],
        "api_key":                    settings.typesense_api_key,
        "connection_timeout_seconds": 10,
    })


def ensure_collection(client: typesense.Client) -> None:
    """Create the collection if it does not exist. Drop and recreate if schema changed."""
    try:
        existing = client.collections[COLLECTION_NAME].retrieve()
        existing_fields = {f["name"] for f in existing.get("fields", [])}
        required_fields = {f["name"] for f in COLLECTION_SCHEMA["fields"]}

        if not required_fields.issubset(existing_fields):
            log.warning("typesense_schema_outdated", collection=COLLECTION_NAME)
            client.collections[COLLECTION_NAME].delete()
            client.collections.create(COLLECTION_SCHEMA)
            log.info("typesense_collection_recreated", collection=COLLECTION_NAME)
        else:
            log.info("typesense_collection_exists", collection=COLLECTION_NAME)

    except ObjectNotFound:
        client.collections.create(COLLECTION_SCHEMA)
        log.info("typesense_collection_created", collection=COLLECTION_NAME)


def upsert_chunks(client: typesense.Client, chunks: list[dict]) -> None:
    if not chunks:
        return
    result = client.collections[COLLECTION_NAME].documents.import_(
        chunks, {"action": "upsert"}
    )
    errors = [r for r in result if not r.get("success")]
    if errors:
        log.warning("typesense_upsert_errors", count=len(errors), sample=errors[:3])
    log.info("typesense_upsert_ok", count=len(chunks) - len(errors))


def mark_chunks_superseded(
    client: typesense.Client,
    document_id: str,
    old_version: int,
) -> None:
    """MOD-04: Mark old version chunks as not latest when a new version is uploaded."""
    try:
        client.collections[COLLECTION_NAME].documents.delete({
            "filter_by": (
                f"document_id:={document_id} && "
                f"doc_version:={old_version} && "
                f"is_latest:=true"
            )
        })
        log.info("typesense_old_version_removed",
                 document_id=document_id, version=old_version)
    except Exception as exc:
        log.warning("typesense_supersede_error", error=str(exc))


def delete_document_chunks(client: typesense.Client, document_id: str) -> int:
    result = client.collections[COLLECTION_NAME].documents.delete(
        {"filter_by": f"document_id:={document_id}"}
    )
    deleted = result.get("num_deleted", 0)
    log.info("typesense_chunks_deleted", document_id=document_id, count=deleted)
    return deleted


def hybrid_search(
    client: typesense.Client,
    query: str,
    embedding: list[float],
    top_k: int = 6,
    filter_by: str | None = None,
) -> list[dict]:
    """
    Hybrid BM25 + vector search with full scope filter support.
    filter_by supports any combination of the extended schema fields.
    """
    params = {
        "q":              query,
        "query_by":       "content",
        "per_page":       top_k,
        "exclude_fields": "embedding",
    }

    if filter_by:
        params["filter_by"] = filter_by

    import json, urllib.request
    settings = get_settings()
    base_url = f"{settings.typesense_protocol}://{settings.typesense_host}:{settings.typesense_port}"

    # Try hybrid via multi_search POST — avoids URL length limit with 3072-dim vectors
    try:
        embedding_str = ",".join(f"{v:.6f}" for v in embedding)
        hybrid_params = {**params, "vector_query": f"embedding:([{embedding_str}], k:{top_k})"}
        payload = json.dumps({"searches": [hybrid_params]}).encode("utf-8")
        url     = f"{base_url}/multi_search?collection={COLLECTION_NAME}"
        req     = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json",
                     "X-TYPESENSE-API-KEY": settings.typesense_api_key},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            hits = data.get("results", [{}])[0].get("hits", [])
            if hits:
                log.info("typesense_hybrid_search_ok", hits=len(hits))
                return [{**h["document"], "score": 1 - h.get("vector_distance", 0.25)} for h in hits]
    except Exception as exc:
        log.warning("typesense_vector_search_failed", error=str(exc))

    # Fallback: keyword-only BM25
    try:
        result = client.collections[COLLECTION_NAME].documents.search(params)
        hits   = result.get("hits", [])
        log.info("typesense_keyword_search_ok", hits=len(hits))
        return [{**h["document"], "score": 0.75} for h in hits]
    except Exception as exc:
        log.error("typesense_search_failed", error=str(exc))
        return []
