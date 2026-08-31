"""The OpenAPI document is a contract; an unintended change fails the build.

Regenerate deliberately with:  uv run python scripts/dump_openapi.py
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

SNAPSHOT = Path(__file__).parent / "openapi_snapshot.json"


def _normalise(schema: dict) -> str:
    return json.dumps(schema, sort_keys=True, indent=2, ensure_ascii=False)


def test_openapi_schema_matches_snapshot(app):
    current = _normalise(app.openapi())

    assert SNAPSHOT.exists(), (
        "OpenAPI snapshot missing. Generate it with: uv run python scripts/dump_openapi.py"
    )
    expected = SNAPSHOT.read_text(encoding="utf-8")

    assert current == expected, (
        "The HTTP contract changed. If intentional, regenerate the snapshot with "
        "`uv run python scripts/dump_openapi.py` and review the diff in the PR."
    )


def test_every_documented_route_is_present(app):
    paths = set(app.openapi()["paths"])
    assert paths == {
        "/healthz",
        "/readyz",
        "/api/v1/chat",
        "/api/v1/chat/stream",
        "/api/v1/documents",
    }


def test_error_envelope_is_documented_on_failure_responses(app):
    schema = app.openapi()
    chat_responses = schema["paths"]["/api/v1/chat"]["post"]["responses"]
    for status in ("401", "500", "503"):
        content = chat_responses[status]["content"]["application/json"]["schema"]
        assert "ErrorResponse" in json.dumps(content)
