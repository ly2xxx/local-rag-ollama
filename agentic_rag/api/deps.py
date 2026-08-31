"""Shared FastAPI dependencies: settings, service, authentication, tenancy."""

import hmac
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, Request

from ..settings import Settings
from .errors import AppError, ErrorCode
from .service import AgentService


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. `tenant_id` is never taken from the request body."""

    api_key_id: str
    tenant_id: str


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_service(request: Request) -> AgentService:
    return request.app.state.service


def get_principal(
    settings: Settings = Depends(get_settings),
    x_api_key: Optional[str] = Header(
        default=None, alias="X-API-Key", description="Tenant API key."
    ),
) -> Principal:
    """Resolves the caller. Fails closed: no configured keys means no access."""
    if not settings.auth_enabled:
        return Principal(api_key_id="anonymous", tenant_id=settings.default_tenant)

    if not x_api_key:
        raise AppError(ErrorCode.UNAUTHENTICATED, "Missing X-API-Key header.")

    # Compared against *every* known key rather than short-circuiting on the
    # first hit, so timing cannot be used to discover a valid prefix.
    matched_tenant: Optional[str] = None
    for known_key, tenant in settings.api_key_map.items():
        if hmac.compare_digest(x_api_key, known_key):
            matched_tenant = tenant

    if matched_tenant is None:
        raise AppError(ErrorCode.UNAUTHENTICATED, "Unknown API key.")

    return Principal(api_key_id=x_api_key[:8], tenant_id=matched_tenant)
