import pytest

pytestmark = pytest.mark.contract


async def test_healthz_is_dependency_free(client, fake_service):
    # Even with every dependency exploding, liveness must answer 200 — otherwise
    # a failing upstream turns into a pod restart loop.
    fake_service.dependencies_error = RuntimeError("redis is on fire")

    response = await client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "agentic-rag-api"
    assert body["version"] == "0.1.0"


async def test_healthz_needs_no_api_key(client):
    assert (await client.get("/healthz")).status_code == 200


async def test_readyz_is_200_when_dependencies_are_healthy(client):
    response = await client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert {d["name"] for d in body["dependencies"]} == {"redis", "vector_store"}


async def test_readyz_reports_503_when_redis_down(client, fake_service):
    fake_service.dependencies = [
        {"name": "redis", "healthy": False, "required": True, "detail": "connection refused"},
        {"name": "vector_store", "healthy": True, "required": True},
    ]

    response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    redis_status = next(d for d in body["dependencies"] if d["name"] == "redis")
    assert redis_status["healthy"] is False
    assert redis_status["detail"] == "connection refused"


async def test_readyz_ignores_unhealthy_optional_dependencies(client, fake_service):
    fake_service.dependencies = [
        {"name": "redis", "healthy": False, "required": False},
        {"name": "vector_store", "healthy": True, "required": True},
    ]

    response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


async def test_readyz_recovers_when_dependency_returns(client, fake_service):
    fake_service.dependencies = [{"name": "redis", "healthy": False, "required": True}]
    assert (await client.get("/readyz")).status_code == 503

    fake_service.dependencies = [{"name": "redis", "healthy": True, "required": True}]
    assert (await client.get("/readyz")).status_code == 200
