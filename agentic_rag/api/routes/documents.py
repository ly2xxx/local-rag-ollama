"""Knowledge-base document endpoints, scoped to the caller's tenant and thread."""

import logging
import os
import re
import tempfile

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status

from ...settings import Settings
from ..deps import Principal, get_principal, get_service, get_settings
from ..errors import AppError, ErrorCode
from ..schemas import (
    THREAD_ID_PATTERN,
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentListResponse,
    ErrorResponse,
)
from ..service import AgentService, new_job_id
from ..tracing import get_trace_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["documents"])

_THREAD_ID_RE = re.compile(THREAD_ID_PATTERN)
_UPLOAD_CHUNK = 1024 * 1024

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}

ThreadIdQuery = Query(
    default="default_session",
    pattern=THREAD_ID_PATTERN,
    description="Conversation scope whose knowledge base is addressed.",
)


def _validate_thread_id(thread_id: str) -> None:
    """Form fields bypass the Pydantic model, so validate explicitly."""
    if not _THREAD_ID_RE.match(thread_id or ""):
        raise AppError(
            ErrorCode.INVALID_REQUEST,
            "thread_id must match [A-Za-z0-9_-]{1,64}.",
            details={"thread_id": thread_id},
        )


def _safe_display_name(filename: str) -> str:
    """Uses the basename only — an upload must never steer a filesystem path."""
    name = os.path.basename(filename or "").strip()
    if not name or not name.lower().endswith(".pdf"):
        raise AppError(ErrorCode.INVALID_REQUEST, "Only .pdf uploads are accepted.")
    return name


async def _spool_to_disk(upload: UploadFile, max_bytes: int) -> str:
    """Streams the upload to a temp file, enforcing the size cap as it goes."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        written = 0
        with tmp:
            while True:
                chunk = await upload.read(_UPLOAD_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise AppError(
                        ErrorCode.PAYLOAD_TOO_LARGE,
                        f"Upload exceeds the {max_bytes} byte limit.",
                    )
                tmp.write(chunk)
        if written == 0:
            raise AppError(ErrorCode.INVALID_REQUEST, "The uploaded file is empty.")
    except Exception:
        os.unlink(tmp.name)
        raise
    return tmp.name


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a PDF into the session knowledge base",
    description="Ingestion is synchronous in Phase 1 but already reports a job_id "
    "and 202, so moving it onto the Phase 5 queue is not a contract change.",
    responses=_ERROR_RESPONSES,
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF document to ingest."),
    thread_id: str = Form(default="default_session"),
    principal: Principal = Depends(get_principal),
    service: AgentService = Depends(get_service),
    settings: Settings = Depends(get_settings),
) -> DocumentIngestResponse:
    _validate_thread_id(thread_id)
    display_name = _safe_display_name(file.filename)
    temp_path = await _spool_to_disk(file, settings.max_upload_bytes)

    try:
        result = await service.ingest(
            tenant_id=principal.tenant_id,
            thread_id=thread_id,
            file_path=temp_path,
            display_name=display_name,
        )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:  # pragma: no cover - best effort cleanup
            logger.warning("Could not remove temp upload %s", temp_path)

    return DocumentIngestResponse(
        job_id=new_job_id(),
        status=result.status,
        document=result.document,
        thread_id=thread_id,
        trace_id=get_trace_id(),
        detail=result.detail,
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List documents in the session knowledge base",
    responses=_ERROR_RESPONSES,
)
async def list_documents(
    thread_id: str = ThreadIdQuery,
    principal: Principal = Depends(get_principal),
    service: AgentService = Depends(get_service),
) -> DocumentListResponse:
    documents = await service.list_documents(
        tenant_id=principal.tenant_id, thread_id=thread_id
    )
    return DocumentListResponse(thread_id=thread_id, documents=documents)


@router.delete(
    "/documents",
    response_model=DocumentDeleteResponse,
    summary="Clear the session knowledge base",
    responses=_ERROR_RESPONSES,
)
async def clear_documents(
    thread_id: str = ThreadIdQuery,
    principal: Principal = Depends(get_principal),
    service: AgentService = Depends(get_service),
) -> DocumentDeleteResponse:
    await service.clear_documents(tenant_id=principal.tenant_id, thread_id=thread_id)
    return DocumentDeleteResponse(thread_id=thread_id, cleared=True)
