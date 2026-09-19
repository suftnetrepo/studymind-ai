"""
Pydantic v2 schemas — API request/response contracts.
Sprint 1: Auth schemas + existing RAG schemas.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ── Auth ───────────────────────────────────────────────────────────────────

VALID_ROLES = {"admin", "lecturer", "student", "self_learner"}


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field(..., min_length=2, max_length=255)
    role: str = Field(..., description="admin | lecturer | student | self_learner")
    institution_code: Optional[str] = Field(default=None, max_length=64)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshRequest(BaseModel):
    refresh_token: str


class UserSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    role: str
    institution_id: Optional[uuid.UUID] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, min_length=2, max_length=255)
    profile: Optional[dict] = None


# ── Institution ────────────────────────────────────────────────────────────

class InstitutionCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    domain: Optional[str] = Field(default=None, max_length=255)
    tier: str = Field(default="free")

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        valid = {"free", "educator", "institution", "enterprise"}
        if v not in valid:
            raise ValueError(f"tier must be one of {valid}")
        return v


class InstitutionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    domain: Optional[str] = None
    tier: str
    is_active: bool
    created_at: datetime


# ── Source citation ────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    content_snippet: str = Field(..., max_length=300)
    relevance_score: float = Field(..., ge=0.0, le=1.0)


# ── Chat ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[uuid.UUID] = None
    message: str = Field(..., min_length=1, max_length=8192)
    stream: bool = False
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: uuid.UUID
    session_id: uuid.UUID
    answer: str
    sources: list[SourceCitation] = []
    latency_ms: int
    token_count: Optional[int] = None


class MessageSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    sources: Optional[list[SourceCitation]] = None
    latency_ms: Optional[int] = None


class SessionCreate(BaseModel):
    title: str = Field(default="New conversation", max_length=255)


class SessionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    message_count: int = 0


class SessionWithMessages(SessionSchema):
    messages: list[MessageSchema] = []


# ── Documents ──────────────────────────────────────────────────────────────

class DocumentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    status: str
    visibility: str
    error_message: Optional[str] = None
    created_at: datetime
    indexed_at: Optional[datetime] = None


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: str
    message: str


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    services: dict[str, bool]


class StatsResponse(BaseModel):
    total_sessions: int
    total_messages: int
    total_documents: int
    total_chunks: int
    total_users: int = 0
    avg_latency_ms: Optional[float] = None


# ── Sprint 2: Institution & Department schemas ─────────────────────────────

class DepartmentCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    code: Optional[str] = Field(default=None, max_length=32)


class DepartmentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    institution_id: uuid.UUID
    name: str
    code: Optional[str] = None
    created_at: datetime


class CreateLecturerRequest(BaseModel):
    """INST-03: Admin creates a lecturer account."""
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=255)
    department_id: Optional[uuid.UUID] = None
    send_invite: bool = True   # email invite — Phase 2


class BulkImportRow(BaseModel):
    """One row from a CSV student import."""
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=255)
    student_id: Optional[str] = Field(default=None, max_length=64)
    department_code: Optional[str] = None


class BulkImportRequest(BaseModel):
    """INST-04: Parsed CSV rows sent to the bulk import endpoint."""
    rows: list[BulkImportRow] = Field(..., min_length=1, max_length=1000)
    department_id: Optional[uuid.UUID] = None
    send_invites: bool = False


class BulkImportResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    total_rows: int
    message: str


class BulkImportStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    total_rows: int
    success_count: int
    error_count: int
    errors: Optional[list] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class InstitutionCodeCreate(BaseModel):
    """INST-05 / INST-06: Generate a join or enrolment code."""
    code_type: str = Field(..., description="institution_join | module_enrolment")
    target_role: str = Field(default="student")
    max_uses: Optional[int] = Field(default=None, ge=1, le=10000)
    expires_days: Optional[int] = Field(default=30, ge=1, le=365)

    @field_validator("code_type")
    @classmethod
    def validate_code_type(cls, v: str) -> str:
        valid = {"institution_join", "module_enrolment"}
        if v not in valid:
            raise ValueError(f"code_type must be one of {valid}")
        return v


class InstitutionCodeSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    code: str
    code_type: str
    target_role: str
    max_uses: Optional[int] = None
    use_count: int
    expires_at: Optional[datetime] = None
    is_active: bool
    created_at: datetime


class JoinWithCodeRequest(BaseModel):
    """INST-05: Student joins institution using a code."""
    code: str = Field(..., min_length=6, max_length=32)


class PasswordResetRequestBody(BaseModel):
    """AUTH-08: Request a password reset email."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """AUTH-08: Submit the reset token and new password."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class AdminUserListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    is_verified: bool
    department_id: Optional[uuid.UUID] = None
    created_at: datetime
    last_login: Optional[datetime] = None


# ── Sprint 3: Modules, Semesters, Weeks, Enrolments ───────────────────────

class SemesterCreate(BaseModel):
    label: str = Field(..., min_length=2, max_length=64,
                       description="e.g. '2025 Semester 1' or 'Autumn 2025'")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_current: bool = True


class SemesterSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    institution_id: Optional[uuid.UUID] = None
    label: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_current: bool
    created_at: datetime


class WeekCreate(BaseModel):
    week_number: int = Field(..., ge=1, le=52)
    title: str = Field(..., min_length=2, max_length=255,
                       description="e.g. 'Week 3 — Binary Trees'")
    description: Optional[str] = Field(default=None, max_length=1000)


class WeekSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    module_id: uuid.UUID
    week_number: int
    title: str
    description: Optional[str] = None
    created_at: datetime


class ModuleCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=255)
    course_code: Optional[str] = Field(default=None, max_length=32,
                                        description="e.g. CSC109 — optional for self-learners")
    description: Optional[str] = Field(default=None, max_length=1000)
    semester_id: Optional[uuid.UUID] = None
    department_id: Optional[uuid.UUID] = None
    access_type: str = Field(default="personal",
                             description="personal | class | institution")
    status: str = Field(default="active",
                        description="active | archived | draft")

    @field_validator("access_type")
    @classmethod
    def validate_access_type(cls, v: str) -> str:
        valid = {"personal", "class", "institution"}
        if v not in valid:
            raise ValueError(f"access_type must be one of {valid}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"active", "archived", "draft"}
        if v not in valid:
            raise ValueError(f"status must be one of {valid}")
        return v


class ModuleSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    course_code: Optional[str] = None
    title: str
    description: Optional[str] = None
    owner_id: uuid.UUID
    institution_id: Optional[uuid.UUID] = None
    department_id: Optional[uuid.UUID] = None
    semester_id: Optional[uuid.UUID] = None
    access_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    student_count: int = 0


class ModuleDetailSchema(ModuleSchema):
    weeks: list[WeekSchema] = []


class EnrolRequest(BaseModel):
    """Student enrolls via enrolment code or direct module ID."""
    enrolment_code: Optional[str] = Field(default=None, max_length=32)
    module_id: Optional[uuid.UUID] = None

    @field_validator("enrolment_code")
    @classmethod
    def validate_code(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v else v


class EnrolmentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    student_id: uuid.UUID
    module_id: uuid.UUID
    enrolled_at: datetime
    status: str


class ModuleDocumentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    module_id: uuid.UUID
    document_id: uuid.UUID
    week_id: Optional[uuid.UUID] = None
    version: int
    is_latest: bool
    visibility: str
    created_at: datetime
    # Joined document fields
    filename: str = ""
    file_type: str = ""
    file_size_bytes: int = 0
    chunk_count: int = 0
    status: str = ""
    indexed_at: Optional[datetime] = None


class DocumentVersionSchema(BaseModel):
    version: int
    created_at: datetime
    is_latest: bool
    document_id: uuid.UUID
    filename: str


# ── Sprint 4: Scoped Chat schemas ──────────────────────────────────────────

class ScopeMode(str):
    """SRCH-02: Scope selector values."""
    EVERYTHING       = "everything"       # class + personal, current semester
    CLASS_ONLY       = "class_only"       # SRCH-09: class materials only
    PERSONAL_ONLY    = "personal_only"    # SRCH-09: personal notes only
    ALL_SEMESTERS    = "all_semesters"    # SRCH-07: includes archived semesters


class ScopedChatRequest(BaseModel):
    """Sprint 4 extended ChatRequest with full scope control."""
    session_id: Optional[uuid.UUID] = None
    module_id:  Optional[uuid.UUID] = None   # QA-06: scope to one module
    message:    str = Field(..., min_length=1, max_length=8192)
    stream:     bool = False
    top_k:      Optional[int] = Field(default=None, ge=1, le=20)

    # SRCH-02: Scope picker
    scope_mode: str = Field(
        default="everything",
        description="everything | class_only | personal_only | all_semesters",
    )

    # SRCH-07: Include previous semesters
    include_archived: bool = False

    # SRCH-03/04: /csc109/week3 shortcut embedded in message is auto-detected
    # No extra field needed — parser reads message prefix

    @field_validator("scope_mode")
    @classmethod
    def validate_scope_mode(cls, v: str) -> str:
        valid = {"everything", "class_only", "personal_only", "all_semesters"}
        if v not in valid:
            raise ValueError(f"scope_mode must be one of {valid}")
        return v


class ScopeIndicator(BaseModel):
    """SRCH-05: Always visible scope context returned with every answer."""
    description: str          # Human-readable: "CSC109 · Week 3 · Semester 2"
    module_id: Optional[str]  = None
    course_code: Optional[str] = None
    semester_label: Optional[str] = None
    week_number: Optional[int] = None
    scope_mode: str = "everything"
    document_count: int = 0


class ScopedChatResponse(BaseModel):
    """Sprint 4 extended ChatResponse with scope indicator."""
    model_config = ConfigDict(from_attributes=True)

    message_id:  uuid.UUID
    session_id:  uuid.UUID
    answer:      str
    sources:     list[SourceCitation] = []
    latency_ms:  int
    token_count: Optional[int] = None
    scope:       Optional[ScopeIndicator] = None   # SRCH-05
    no_content_found: bool = False                  # SRCH-06: true when nothing retrieved
    suggestions: list[str] = []                     # SRCH-06: scope widening suggestions


class SessionWithScope(BaseModel):
    """Chat session with its module scope."""
    model_config = ConfigDict(from_attributes=True)

    id:        uuid.UUID
    title:     str
    module_id: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime
    is_active: bool
    message_count: int = 0


# ── Sprint 5: Quiz schemas ─────────────────────────────────────────────────

class QuizGenerateRequest(BaseModel):
    module_id:      Optional[uuid.UUID] = None
    week_id:        Optional[uuid.UUID] = None
    document_id:    Optional[uuid.UUID] = None
    question_count: int  = Field(default=10, ge=5, le=30)
    question_type:  str  = Field(default="mcq",
                                 description="mcq | short_answer | true_false")
    title:          Optional[str] = Field(default=None, max_length=255)
    topic:          Optional[str] = Field(default=None, max_length=200,
                                          description="Topic to focus on, e.g. 'Python data types'")

    @field_validator("question_type")
    @classmethod
    def validate_qtype(cls, v: str) -> str:
        valid = {"mcq", "short_answer", "true_false"}
        if v not in valid:
            raise ValueError(f"question_type must be one of {valid}")
        return v


class QuizOptionSchema(BaseModel):
    id:   str
    text: str


class QuizQuestionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:             uuid.UUID
    position:       int
    question:       str
    options:        Optional[list[QuizOptionSchema]] = None
    correct_answer: str
    explanation:    str
    source_chunk:   Optional[str] = None
    student_answer: Optional[str] = None
    is_correct:     Optional[bool] = None


class QuizAttemptSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:             uuid.UUID
    module_id:      Optional[uuid.UUID] = None
    week_id:        Optional[uuid.UUID] = None
    title:          str
    question_type:  str
    question_count: int
    score:          Optional[float] = None
    status:         str
    created_at:     datetime
    submitted_at:   Optional[datetime] = None
    questions:      list[QuizQuestionSchema] = []


class QuizSubmitAnswer(BaseModel):
    question_id: uuid.UUID
    answer:      str = Field(..., min_length=1)


class QuizSubmitRequest(BaseModel):
    answers: list[QuizSubmitAnswer] = Field(..., min_length=1)


class QuizSubmitResponse(BaseModel):
    attempt_id:    uuid.UUID
    score:         float
    total:         int
    correct:       int
    questions:     list[QuizQuestionSchema]


# ── Sprint 5: Flashcard schemas ────────────────────────────────────────────

class FlashcardGenerateRequest(BaseModel):
    module_id:   Optional[uuid.UUID] = None
    week_id:     Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    title:       Optional[str] = Field(default=None, max_length=255)
    max_cards:   int = Field(default=20, ge=5, le=50)
    topic:       Optional[str] = Field(default=None, max_length=200,
                                       description="Topic to focus on, e.g. 'Python data types'")


class FlashcardSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:           uuid.UUID
    position:     int
    front:        str
    back:         str
    source_chunk: Optional[str] = None
    status:       str


class FlashcardDeckSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:              uuid.UUID
    module_id:       Optional[uuid.UUID] = None
    week_id:         Optional[uuid.UUID] = None
    title:           str
    card_count:      int
    mastered_count:  int
    created_at:      datetime
    last_studied_at: Optional[datetime] = None
    cards:           list[FlashcardSchema] = []


class FlashcardUpdateRequest(BaseModel):
    status: str = Field(..., description="new | learning | mastered")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"new", "learning", "mastered"}
        if v not in valid:
            raise ValueError(f"status must be one of {valid}")
        return v


# ── Sprint 5: Summary schemas ──────────────────────────────────────────────

class SummariseRequest(BaseModel):
    module_id:   Optional[uuid.UUID] = None
    week_id:     Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    scope:       str = Field(
        default="document",
        description="document | week | module"
    )
    topic:       Optional[str] = Field(default=None, max_length=200,
                                       description="Topic to focus on, e.g. 'Python data types'")

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v: str) -> str:
        valid = {"document", "week", "module"}
        if v not in valid:
            raise ValueError(f"scope must be one of {valid}")
        return v


class SummarySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:               uuid.UUID
    module_id:        Optional[uuid.UUID] = None
    week_id:          Optional[uuid.UUID] = None
    document_id:      Optional[uuid.UUID] = None
    scope:            str
    content:          str
    source_doc_count: int
    created_at:       datetime
