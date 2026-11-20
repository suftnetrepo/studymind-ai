"""
AI grading pipeline for student assignment submissions.

Flow:
  1. Download assignment brief + marking scheme from Cloudinary (private signed URLs).
  2. Download student submission from Cloudinary.
  3. Extract text from PDF or Word (.docx) file for all three documents.
  4. Send brief + marking scheme + student work to GPT for structured grading.
  5. Return GradingResult with score (%), strengths, weaknesses, recommendations.

The marking scheme is never exposed outside this module.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import Optional

import httpx
from openai import AsyncOpenAI

from app.config import get_settings
from app.logging_config import get_logger

log = get_logger(__name__)

# ── Text extraction ────────────────────────────────────────────────────────

def _extract_pdf_text(raw: bytes) -> str:
    """Extract plain text from PDF bytes using pypdf."""
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(io.BytesIO(raw))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
        return "\n\n".join(parts)
    except Exception as e:
        log.warning("pdf_extraction_failed", error=str(e))
        return ""


def _extract_docx_text(raw: bytes) -> str:
    """Extract plain text from Word .docx bytes using python-docx."""
    try:
        import docx  # type: ignore
        doc = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        log.warning("docx_extraction_failed", error=str(e))
        return ""


def extract_text(raw: bytes, filename: str) -> str:
    """Route extraction by file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return _extract_pdf_text(raw)
    if ext in ("docx", "doc"):
        return _extract_docx_text(raw)
    # Fallback: treat as UTF-8 text
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


# ── File download ──────────────────────────────────────────────────────────

async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url, follow_redirects=True)
        r.raise_for_status()
        return r.content


# ── Grading result ─────────────────────────────────────────────────────────

@dataclass
class GradingResult:
    score: float                   # 0–100 percentage
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    notes: str                     # raw reasoning / fallback message


# ── Main grading function ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a fair, rigorous academic grader. You will receive:
1. The assignment brief (what students were asked to do).
2. The lecturer's marking scheme (the private grading criteria).
3. The student's submitted work.

Your task is to grade the submission against the marking scheme and produce a JSON object with:
- "score": integer 0–100 representing the percentage earned
- "strengths": array of 2–5 short strings (what the student did well)
- "weaknesses": array of 2–5 short strings (gaps, errors, or missed criteria)
- "recommendations": array of 2–4 short strings (concrete suggestions for improvement)
- "notes": a brief string summarising your reasoning (2–3 sentences)

Respond ONLY with valid JSON — no markdown fences, no preamble."""

_MAX_CHARS = 12_000  # per document, to stay within context limits


def _truncate(text: str, label: str) -> str:
    if len(text) <= _MAX_CHARS:
        return text
    log.info("grading_text_truncated", label=label, original=len(text), limit=_MAX_CHARS)
    return text[:_MAX_CHARS] + f"\n[... truncated — {len(text) - _MAX_CHARS} chars omitted]"


async def grade_submission(
    *,
    brief_url: str,
    brief_filename: str,
    scheme_url: str,
    scheme_filename: str,
    submission_url: str,
    submission_filename: str,
) -> GradingResult:
    """
    Download all three files, extract their text, and ask GPT to grade.
    Returns a GradingResult. Never raises — on failure returns score=0 with
    an error note so the submission can be marked 'error' cleanly.
    """
    settings = get_settings()

    try:
        brief_bytes, scheme_bytes, work_bytes = await _parallel_downloads(
            brief_url, scheme_url, submission_url
        )
    except Exception as e:
        log.error("grading_download_failed", error=str(e))
        return GradingResult(
            score=0,
            strengths=[],
            weaknesses=[],
            recommendations=[],
            notes=f"Download failed: {e}",
        )

    brief_text    = _truncate(extract_text(brief_bytes,  brief_filename),    "brief")
    scheme_text   = _truncate(extract_text(scheme_bytes, scheme_filename),   "scheme")
    work_text     = _truncate(extract_text(work_bytes,   submission_filename), "submission")

    if not work_text.strip():
        return GradingResult(
            score=0,
            strengths=[],
            weaknesses=["Could not extract text from submission — file may be corrupt or unreadable."],
            recommendations=["Resubmit as a valid PDF or Word document containing selectable text."],
            notes="Text extraction returned empty content.",
        )

    user_msg = (
        f"=== ASSIGNMENT BRIEF ===\n{brief_text}\n\n"
        f"=== MARKING SCHEME ===\n{scheme_text}\n\n"
        f"=== STUDENT SUBMISSION ===\n{work_text}"
    )

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw_json = response.choices[0].message.content or "{}"
        data = json.loads(raw_json)
    except Exception as e:
        log.error("grading_llm_failed", error=str(e))
        return GradingResult(
            score=0,
            strengths=[],
            weaknesses=[],
            recommendations=[],
            notes=f"Grading AI error: {e}",
        )

    # Safely parse fields — GPT may omit or mis-type some
    score = _clamp_score(data.get("score", 0))
    return GradingResult(
        score=score,
        strengths=_to_str_list(data.get("strengths", [])),
        weaknesses=_to_str_list(data.get("weaknesses", [])),
        recommendations=_to_str_list(data.get("recommendations", [])),
        notes=str(data.get("notes", "")),
    )


# ── Helpers ────────────────────────────────────────────────────────────────

import asyncio


async def _parallel_downloads(*urls: str) -> list[bytes]:
    return list(await asyncio.gather(*(_download(u) for u in urls)))


def _clamp_score(raw: object) -> float:
    try:
        v = float(raw)  # type: ignore[arg-type]
        return max(0.0, min(100.0, v))
    except (TypeError, ValueError):
        return 0.0


def _to_str_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []
