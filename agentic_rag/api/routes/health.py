"""Liveness and readiness probes. Unauthenticated by design — kubelet has no key."""

from fastapi import APIRouter, Depends, Response, status

from ...settings import Settings
from ..deps import get_service, get_settings
from ..schemas import DependencyStatus, HealthResponse, ReadyResponse
from ..service import AgentService

router = APIRouter(tags=["health"])


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Answers from the process alone. Never touches a dependency, so a "
    "failing upstream can never trigger a restart loop.",
)
async def healthz(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok", service=settings.app_name, version=settings.app_version
    )


@router.get(
    "/readyz",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description="Live-probes every backing service. Returns 503 while a required "
    "dependency is unreachable so the pod leaves the load-balancer rotation.",
    responses={503: {"model": ReadyResponse}},
)
async def readyz(
    response: Response, service: AgentService = Depends(get_service)
) -> ReadyResponse:
    checks = await service.check_dependencies()
    dependencies = [DependencyStatus(**check) for check in checks]

    ready = all(dep.healthy for dep in dependencies if dep.required)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadyResponse(
        status="ready" if ready else "not_ready", dependencies=dependencies
    )
