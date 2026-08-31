"""Regenerates the OpenAPI contract snapshot.

    uv run python scripts/dump_openapi.py

Run this deliberately when the HTTP contract changes, and review the diff in the
PR — `tests/contract/test_openapi_snapshot.py` fails until it is updated.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag.api.main import create_app  # noqa: E402
from agentic_rag.settings import Settings  # noqa: E402

SNAPSHOT = PROJECT_ROOT / "tests" / "contract" / "openapi_snapshot.json"


def main() -> int:
    # Same settings the test fixture uses, so the snapshot is reproducible.
    settings = Settings(
        _env_file=None,
        app_version="0.1.0",
        api_keys="key-tenant-a:tenant_a,key-tenant-b:tenant_b",
        auth_enabled=True,
        warm_embeddings_on_startup=False,
    )
    app = create_app(settings=settings, service=object())
    schema = json.dumps(app.openapi(), sort_keys=True, indent=2, ensure_ascii=False)

    previous = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else None
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(schema, encoding="utf-8")

    relative = SNAPSHOT.relative_to(PROJECT_ROOT)
    if previous is None:
        print(f"Created {relative}")
    elif previous == schema:
        print(f"{relative} is unchanged.")
    else:
        print(f"Updated {relative} — review the diff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
