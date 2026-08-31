import pytest

from agentic_rag.api.errors import (
    ERROR_SPECS,
    AppError,
    ErrorCode,
    code_for_status,
    retryable_for,
    status_for,
)

pytestmark = pytest.mark.unit

# The table in DESIGN.md §5.4, restated here so a drift in either direction fails.
EXPECTED = {
    ErrorCode.INVALID_REQUEST: (400, False),
    ErrorCode.UNAUTHENTICATED: (401, False),
    ErrorCode.NOT_FOUND: (404, False),
    ErrorCode.PAYLOAD_TOO_LARGE: (413, False),
    ErrorCode.GUARDRAIL_BLOCKED: (422, False),
    ErrorCode.RATE_LIMITED: (429, True),
    ErrorCode.RETRIEVAL_UNAVAILABLE: (503, True),
    ErrorCode.LLM_UNAVAILABLE: (503, True),
    ErrorCode.LLM_TIMEOUT: (504, True),
    ErrorCode.INTERNAL: (500, False),
}


@pytest.mark.parametrize("code,expected", EXPECTED.items(), ids=lambda v: getattr(v, "value", ""))
def test_each_taxonomy_code_maps_to_expected_status(code, expected):
    status, retryable = expected
    assert status_for(code) == status
    assert retryable_for(code) is retryable


def test_taxonomy_has_no_undocumented_codes():
    assert set(ERROR_SPECS) == set(EXPECTED)


def test_app_error_payload_shape():
    error = AppError(ErrorCode.RATE_LIMITED)
    payload = error.to_payload("trace-1")["error"]
    assert payload == {
        "code": "RATE_LIMITED",
        "message": "Rate limit exceeded.",
        "trace_id": "trace-1",
        "retryable": True,
    }


def test_app_error_details_are_included_when_present():
    error = AppError(ErrorCode.INVALID_REQUEST, "bad", details={"field": "query"})
    assert error.to_payload("t")["error"]["details"] == {"field": "query"}


def test_retryable_can_be_overridden_explicitly():
    assert AppError(ErrorCode.LLM_UNAVAILABLE, retryable=False).retryable is False


@pytest.mark.parametrize(
    "status,expected",
    [(404, ErrorCode.NOT_FOUND), (405, ErrorCode.INTERNAL), (422, ErrorCode.INVALID_REQUEST)],
)
def test_framework_statuses_map_back_into_the_taxonomy(status, expected):
    assert code_for_status(status) is expected


async def test_internal_error_never_leaks_traceback(app, client, auth_a, fake_service):
    secret = "sk-live-do-not-leak"
    fake_service.error = ValueError(f"boom {secret} at C:/internal/path.py")

    response = await client.post(
        "/api/v1/chat", json={"query": "hello"}, headers=auth_a
    )

    assert response.status_code == 500
    body = response.text
    assert secret not in body
    assert "Traceback" not in body
    assert "path.py" not in body

    payload = response.json()["error"]
    assert payload["code"] == "INTERNAL"
    assert payload["retryable"] is False
    assert payload["trace_id"]
