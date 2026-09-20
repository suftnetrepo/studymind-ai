"""Writing Assistant: general, outline, essay and modify modes."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.auth.dependencies import require_auth
from app.config import get_settings
from app.db.models import User
from app.logging_config import get_logger

log    = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["writing"])


class WritingRequest(BaseModel):
    mode:        str = Field(..., pattern="^(general|outline|essay|modify)$")
    input:       str = Field(..., min_length=1, max_length=5000)
    format:      Optional[str] = Field(default=None, max_length=60)
    tone:        Optional[str] = Field(default=None, max_length=60)
    essay_type:  Optional[str] = Field(default=None, max_length=60)
    level:       Optional[str] = Field(default=None, max_length=60)
    length:      Optional[str] = Field(default=None, max_length=60)
    paragraphs:  Optional[str] = Field(default=None, max_length=3)
    modify_type: Optional[str] = Field(default=None, max_length=60)


MODIFY_INSTRUCTIONS = {
    "Expand":           "Make the text longer with more detail and examples.",
    "Shorten":          "Make the text more concise without losing meaning.",
    "Continue Writing": "Continue the text naturally from where it ends. Return only the continuation.",
    "Improve Grammar":  "Fix all grammar, punctuation and style issues.",
    "Make Formal":      "Rewrite in a formal, professional tone.",
    "Simplify":         "Rewrite in simpler, clearer language.",
}


def _build_prompt(r: WritingRequest) -> str:
    if r.mode == "general":
        return (
            "Write content based on the following request.\n"
            f"Format: {r.format or 'General'}\nTone: {r.tone or 'Auto'}\n"
            f"Topic/Request: {r.input}\n"
            "Produce well-structured, high-quality written content."
        )
    if r.mode == "outline":
        return (
            "Create a detailed, well-structured outline for the following topic. "
            "Include main sections with sub-points, as a clear hierarchical outline.\n"
            f"Topic: {r.input}"
        )
    if r.mode == "essay":
        return (
            f"Write a complete {r.essay_type or 'Expository'} essay on the following topic.\n"
            f"Proficiency level: {r.level or 'Intermediate'}\n"
            f"Target length: {r.length or 'Medium (600 words)'}\n"
            f"Number of paragraphs: {r.paragraphs or '5'}\n"
            f"Topic: {r.input}\n"
            "Include an introduction, body paragraphs and a conclusion, using academic "
            "language appropriate to the proficiency level."
        )
    kind = r.modify_type or "Expand"
    instruction = MODIFY_INSTRUCTIONS.get(kind, MODIFY_INSTRUCTIONS["Expand"])
    return (
        f"Modification type: {kind}. {instruction}\n"
        "Apply it to the text below. Return only the resulting text, no explanations.\n\n"
        f"Text:\n{r.input}"
    )


@router.post("/writing/generate")
async def generate_writing(
    req: WritingRequest,
    current_user: User = Depends(require_auth),
):
    settings = get_settings()
    try:
        resp = await AsyncOpenAI(api_key=settings.openai_api_key).chat.completions.create(
            model=settings.openai_chat_model,
            max_tokens=2500,
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are StudyMind AI's writing assistant. Produce high quality written content as requested. "
                 "Format with Markdown (headings, bold, lists, tables where useful). For mathematics and science, "
                 "write symbols directly in Unicode (x², H₂O, √, π, ∑, →, ≤, Δ, ±, ×, °) and NEVER use LaTeX or dollar-sign delimiters."},
                {"role": "user",   "content": _build_prompt(req)},
            ],
        )
    except Exception as exc:
        log.error("writing_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="The AI service is unavailable. Try again.")
    return {"content": (resp.choices[0].message.content or "").strip()}
