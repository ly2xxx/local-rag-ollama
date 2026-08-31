"""Error taxonomy (DESIGN.md §5.4).

Every failure leaving the API is one of these codes. The client sees
`{"error": {"code", "message", "trace_id", "retryable"}}` and never a stack trace.
"""

from enum import Enum
from typing import Any, Dict, NamedTuple, Optional


class ErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    NOT_FOUND = "NOT_FOUND"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    RATE_LIMITED = "RATE_LIMITED"
    GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
    RETRIEVAL_UNAVAILABLE = "RETRIEVAL_UNAVAILABLE"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    INTERNAL = "INTERNAL"


class ErrorSpec(NamedTuple):
    status: int
    retryable: bool
    message: str


ERROR_SPECS: Dict[ErrorCode, ErrorSpec] = {
    ErrorCode.INVALID_REQUEST: ErrorSpec(400, False, "The request could not be validated."),
    ErrorCode.UNAUTHENTICATED: ErrorSpec(401, False, "A valid API key is required."),
    ErrorCode.NOT_FOUND: ErrorSpec(404, False, "The requested resource does not exist."),
    ErrorCode.PAYLOAD_TOO_LARGE: ErrorSpec(413, False, "The uploaded payload is too large."),
    ErrorCode.RATE_LIMITED: ErrorSpec(429, True, "Rate limit exceeded."),
    ErrorCode.GUARDRAIL_BLOCKED: ErrorSpec(422, False, "The request was blocked by a safety policy."),
    ErrorCode.RETRIEVAL_UNAVAILABLE: ErrorSpec(503, True, "The document store is unavailable."),
    ErrorCode.LLM_UNAVAILABLE: ErrorSpec(503, True, "The language model is unavailable."),
    ErrorCode.LLM_TIMEOUT: ErrorSpec(504, True, "The language model timed out."),
    ErrorCode.INTERNAL: ErrorSpec(500, False, "An internal error occurred."),
}


def status_for(code: ErrorCode) -> int:
    return ERROR_SPECS[code].status


def retryable_for(code: ErrorCode) -> bool:
    return ERROR_SPECS[code].retryable


def default_message_for(code: ErrorCode) -> str:
    return ERROR_SPECS[code].message


def error_body(
    code: ErrorCode,
    message: str,
    trace_id: str,
    *,
    retryable: Optional[bool] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The inner error object — also the payload of the SSE `error` event."""
    body: Dict[str, Any] = {
        "code": code.value,
        "message": message,
        "trace_id": trace_id,
        "retryable": retryable_for(code) if retryable is None else retryable,
    }
    if details:
        body["details"] = details
    return body


def error_payload(
    code: ErrorCode,
    message: str,
    trace_id: str,
    *,
    retryable: Optional[bool] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The JSON body of every error response."""
    return {"error": error_body(code, message, trace_id, retryable=retryable, details=details)}


class AppError(Exception):
    """A failure with a known place in the taxonomy."""

    def __init__(
        self,
        code: ErrorCode,
        message: Optional[str] = None,
        *,
        retryable: Optional[bool] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message or default_message_for(code)
        self.status_code = status_for(code)
        self.retryable = retryable_for(code) if retryable is None else retryable
        self.details = details or {}
        super().__init__(self.message)

    def to_payload(self, trace_id: str) -> Dict[str, Any]:
        return error_payload(
            self.code,
            self.message,
            trace_id,
            retryable=self.retryable,
            details=self.details,
        )


# Status codes raised by the framework itself (404 on an unknown route, 405, ...)
# mapped back into the taxonomy so every response shares one envelope.
_STATUS_TO_CODE = {
    400: ErrorCode.INVALID_REQUEST,
    401: ErrorCode.UNAUTHENTICATED,
    404: ErrorCode.NOT_FOUND,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    422: ErrorCode.INVALID_REQUEST,
    429: ErrorCode.RATE_LIMITED,
    503: ErrorCode.RETRIEVAL_UNAVAILABLE,
    504: ErrorCode.LLM_TIMEOUT,
}


def code_for_status(status_code: int) -> ErrorCode:
    return _STATUS_TO_CODE.get(status_code, ErrorCode.INTERNAL)
