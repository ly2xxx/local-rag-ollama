"""Shared fixtures and the determinism guards from DESIGN.md §6.3."""

import socket
import sys
import threading
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag.api.main import create_app  # noqa: E402
from agentic_rag.settings import Settings  # noqa: E402
from tests.fakes.service import FakeAgentService  # noqa: E402

TENANT_A_KEY = "key-tenant-a"
TENANT_B_KEY = "key-tenant-b"


def pytest_configure(config):
    for marker, description in [
        ("unit", "Pure logic; no I/O, no network."),
        ("contract", "HTTP surface via ASGI transport; no real upstreams."),
        ("integration", "Needs a real Redis / vector store."),
        ("eval", "Needs a real LLM; slow. Opt-in."),
        ("e2e", "Runs against a deployed cluster. Opt-in."),
        ("allow_network", "Opts out of the outbound-connection guard."),
    ]:
        config.addinivalue_line("markers", f"{marker}: {description}")


# Settings fields whose value changes what a test *means*. `config.py` calls
# load_dotenv() at import, which pushes the developer's .env into os.environ —
# and `Settings(_env_file=None)` still reads os.environ, so `_env_file=None`
# alone does NOT make settings hermetic. A local `AUTH_ENABLED=false` silently
# collapsed every tenant to one, which made a tenant-isolation test "fail"
# against perfectly correct code. These are cleared for every test; upstream
# addresses (REDIS_URL, OLLAMA_*) are left alone because integration tests need
# to reach the real services.
_BEHAVIOURAL_ENV_VARS = (
    "APP_NAME",
    "APP_VERSION",
    "ENVIRONMENT",
    "API_KEYS",
    "AUTH_ENABLED",
    "DEFAULT_TENANT",
    "MAX_UPLOAD_BYTES",
    "MAX_QUERY_CHARS",
    "REQUEST_TIMEOUT_SECONDS",
    "SSE_HEARTBEAT_SECONDS",
    "SSE_TOKEN_CHUNK_CHARS",
    "SSE_OBSERVATION_PREVIEW_BYTES",
    "READYZ_REQUIRE_REDIS",
    "REDIS_PING_TIMEOUT_SECONDS",
    "WARM_EMBEDDINGS_ON_STARTUP",
    "CORS_ALLOW_ORIGINS",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def hermetic_settings_env(monkeypatch):
    """Stops the developer's .env from reconfiguring the suite."""
    for name in _BEHAVIOURAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_CREATE_CONNECTION = socket.create_connection
_REAL_SOCKETPAIR = socket.socketpair

# asyncio's ProactorEventLoop (Windows) builds its self-pipe with
# socket.socketpair(), which falls back to a real loopback connect. That is
# machinery, not the test reaching out, so it is exempted while it runs.
_internal = threading.local()


@pytest.fixture(autouse=True)
def block_outbound_network(request, monkeypatch):
    """Fails any test that quietly reaches the network.

    Integration/eval/e2e tests opt out; everything else must be hermetic or the
    suite stops being trustworthy.
    """
    exempt = {"integration", "eval", "e2e", "allow_network"}
    if exempt & {marker.name for marker in request.node.iter_markers()}:
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "Outbound network access is blocked in this test. Mark it "
            "@pytest.mark.integration or @pytest.mark.allow_network if intended."
        )

    def _guard(real):
        def wrapper(*args, **kwargs):
            if getattr(_internal, "active", False):
                return real(*args, **kwargs)
            return _blocked(*args, **kwargs)

        return wrapper

    def _guarded_socketpair(*args, **kwargs):
        _internal.active = True
        try:
            return _REAL_SOCKETPAIR(*args, **kwargs)
        finally:
            _internal.active = False

    monkeypatch.setattr(socket.socket, "connect", _guard(_REAL_CONNECT))
    monkeypatch.setattr(socket.socket, "connect_ex", _guard(_REAL_CONNECT_EX))
    monkeypatch.setattr(socket, "create_connection", _guard(_REAL_CREATE_CONNECTION))
    monkeypatch.setattr(socket, "socketpair", _guarded_socketpair)


@pytest.fixture
def settings() -> Settings:
    """Hermetic settings; `_env_file=None` keeps the developer's .env out of tests."""
    return Settings(
        _env_file=None,
        app_version="0.1.0",
        api_keys=f"{TENANT_A_KEY}:tenant_a,{TENANT_B_KEY}:tenant_b",
        auth_enabled=True,
        warm_embeddings_on_startup=False,
        sse_heartbeat_seconds=15.0,
        sse_token_chunk_chars=8,
        max_upload_bytes=1024 * 1024,
        cors_allow_origins="*",
    )


@pytest.fixture
def fake_service() -> FakeAgentService:
    return FakeAgentService()


@pytest.fixture
def app(settings, fake_service):
    return create_app(settings=settings, service=fake_service)


@pytest.fixture
async def client(app):
    """`raise_app_exceptions=False` so unhandled-error responses are observable."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_a() -> dict:
    return {"X-API-Key": TENANT_A_KEY}


@pytest.fixture
def auth_b() -> dict:
    return {"X-API-Key": TENANT_B_KEY}


@pytest.fixture
def sample_pdf() -> Path:
    pdf = PROJECT_ROOT / "cidr2021_paper17.pdf"
    if not pdf.exists():
        pytest.skip("sample PDF not present in the repository")
    return pdf
