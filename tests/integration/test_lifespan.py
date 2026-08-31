"""Integration tests: real Redis and a real embedding model.

Run with:  uv run pytest -m integration
Redis is expected at REDIS_URL (default redis://localhost:6379); tests skip if
it is unreachable rather than failing the developer's inner loop.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from agentic_rag.api.main import create_app
from agentic_rag.api.service import AgenticRAGService
from agentic_rag.settings import Settings
from tests.fakes.service import FakeHelper

pytestmark = pytest.mark.integration

DEAD_REDIS_URL = "redis://127.0.0.1:6399"


def _settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        api_keys="key-tenant-a:tenant_a",
        auth_enabled=True,
        warm_embeddings_on_startup=False,
    )
    base.update(overrides)
    return Settings(**base)


def _redis_available(url: str) -> bool:
    import redis

    try:
        redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=1.0).ping()
        return True
    except Exception:
        return False


@pytest.fixture
def live_redis_url():
    url = Settings(_env_file=None).redis_url
    if not _redis_available(url):
        pytest.skip(f"Redis not reachable at {url}")
    return url


async def test_readyz_ok_against_live_redis(live_redis_url):
    settings = _settings(redis_url=live_redis_url)
    service = AgenticRAGService(settings, helper=FakeHelper())
    app = create_app(settings=settings, service=service)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    redis_dep = next(d for d in response.json()["dependencies"] if d["name"] == "redis")
    assert redis_dep["healthy"] is True


async def test_readyz_detects_unreachable_redis():
    """Stands in for a Redis bounce: the probe is live, not a cached startup value."""
    settings = _settings(redis_url=DEAD_REDIS_URL, redis_ping_timeout_seconds=0.5)
    service = AgenticRAGService(settings, helper=FakeHelper())
    app = create_app(settings=settings, service=service)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    redis_dep = next(d for d in body["dependencies"] if d["name"] == "redis")
    assert redis_dep["healthy"] is False
    assert redis_dep["detail"]


async def test_readyz_tolerates_redis_when_not_required():
    settings = _settings(
        redis_url=DEAD_REDIS_URL, redis_ping_timeout_seconds=0.5, readyz_require_redis=False
    )
    service = AgenticRAGService(settings, helper=FakeHelper())
    app = create_app(settings=settings, service=service)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/readyz")

    assert response.status_code == 200


async def test_embedding_model_loaded_once_across_requests():
    """The shared cache is what keeps a per-tenant store from costing 90 MB each."""
    from agentic_rag.tools import get_embeddings

    first = get_embeddings()
    second = get_embeddings()

    assert first is second


async def test_warm_up_populates_the_embedding_cache():
    settings = _settings(warm_embeddings_on_startup=True)
    service = AgenticRAGService(settings, helper=FakeHelper())

    await service.warm_up()

    from agentic_rag.tools import get_embeddings

    assert get_embeddings() is get_embeddings()


async def test_lifespan_constructs_a_service_when_none_is_injected(live_redis_url, monkeypatch):
    """The production path: create_app() with no service builds one on startup."""
    constructed = {}

    class StubService:
        async def warm_up(self):
            constructed["warmed"] = True

        async def check_dependencies(self):
            return [{"name": "redis", "healthy": True, "required": True}]

    monkeypatch.setattr(
        "agentic_rag.api.main.AgenticRAGService", lambda settings: StubService()
    )
    app = create_app(settings=_settings(redis_url=live_redis_url))
    assert app.state.service is None

    # ASGITransport does not run the lifespan, so enter it explicitly.
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.service, StubService)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/readyz")

    assert response.status_code == 200
    assert constructed.get("warmed") is True
