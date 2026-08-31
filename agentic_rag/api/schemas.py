"""Pydantic v2 request/response models for the API tier."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

# Shared field definition so every endpoint validates thread_id identically.
ThreadIdField = Field(
    default="default_session",
    pattern=THREAD_ID_PATTERN,
    description="Conversation scope. Alphanumerics, underscore and hyphen only.",
    examples=["session_1"],
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=10_000, description="The user's question.")
    thread_id: str = ThreadIdField

    @field_validator("query")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class Citation(BaseModel):
    """DESIGN.md §5.3. Populated from Phase 3; empty until then."""

    n: int = Field(ge=1)
    chunk_id: str
    source: str
    page: Optional[int] = None
    score: Optional[float] = None


class ScratchpadEntry(BaseModel):
    type: Literal["tool_call", "tool_observation"]
    name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    id: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    trace_id: str
    citations: List[Citation] = Field(default_factory=list)
    scratchpad: List[ScratchpadEntry] = Field(default_factory=list)
    usage: Optional[Usage] = None
    latency_ms: int
    degraded: List[str] = Field(default_factory=list)


class DocumentIngestResponse(BaseModel):
    job_id: str
    status: Literal["completed", "queued", "failed"]
    document: str
    thread_id: str
    trace_id: str
    detail: str


class DocumentListResponse(BaseModel):
    thread_id: str
    documents: List[str]


class DocumentDeleteResponse(BaseModel):
    thread_id: str
    cleared: bool


class DependencyStatus(BaseModel):
    name: str
    healthy: bool
    required: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    dependencies: List[DependencyStatus]


class ErrorBody(BaseModel):
    code: str
    message: str
    trace_id: str
    retryable: bool
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
