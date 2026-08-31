import pytest

pytestmark = pytest.mark.contract


async def test_ingest_returns_202_with_job_id(client, auth_a, sample_pdf):
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("paper.pdf", sample_pdf.read_bytes(), "application/pdf")},
        data={"thread_id": "session_1"},
        headers=auth_a,
    )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["status"] == "completed"
    assert body["document"] == "paper.pdf"
    assert body["thread_id"] == "session_1"


async def test_ingest_requires_authentication(client, sample_pdf):
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("paper.pdf", sample_pdf.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 401


async def test_non_pdf_upload_is_rejected(client, auth_a):
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        headers=auth_a,
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


async def test_empty_upload_is_rejected(client, auth_a):
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        headers=auth_a,
    )
    assert response.status_code == 400


async def test_oversized_upload_is_413(client, auth_a, settings):
    oversized = b"x" * (settings.max_upload_bytes + 1024)

    response = await client.post(
        "/api/v1/documents",
        files={"file": ("big.pdf", oversized, "application/pdf")},
        headers=auth_a,
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


async def test_filename_path_components_are_stripped(client, auth_a, sample_pdf, fake_service):
    await client.post(
        "/api/v1/documents",
        files={"file": ("../../evil.pdf", sample_pdf.read_bytes(), "application/pdf")},
        data={"thread_id": "session_1"},
        headers=auth_a,
    )

    stored = fake_service.documents["tenant_a::session_1"]
    assert stored == ["evil.pdf"]


async def test_bad_thread_id_form_field_is_rejected(client, auth_a, sample_pdf):
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("paper.pdf", sample_pdf.read_bytes(), "application/pdf")},
        data={"thread_id": "../../etc"},
        headers=auth_a,
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


async def test_list_documents_is_scoped_to_thread(client, auth_a, sample_pdf):
    pdf = sample_pdf.read_bytes()
    await client.post(
        "/api/v1/documents",
        files={"file": ("one.pdf", pdf, "application/pdf")},
        data={"thread_id": "thread_one"},
        headers=auth_a,
    )

    listed = await client.get(
        "/api/v1/documents", params={"thread_id": "thread_one"}, headers=auth_a
    )
    other = await client.get(
        "/api/v1/documents", params={"thread_id": "thread_two"}, headers=auth_a
    )

    assert listed.json()["documents"] == ["one.pdf"]
    assert other.json()["documents"] == []


async def test_delete_clears_only_that_thread(client, auth_a, sample_pdf):
    pdf = sample_pdf.read_bytes()
    for thread in ("thread_one", "thread_two"):
        await client.post(
            "/api/v1/documents",
            files={"file": (f"{thread}.pdf", pdf, "application/pdf")},
            data={"thread_id": thread},
            headers=auth_a,
        )

    deleted = await client.delete(
        "/api/v1/documents", params={"thread_id": "thread_one"}, headers=auth_a
    )

    assert deleted.status_code == 200
    assert deleted.json() == {"thread_id": "thread_one", "cleared": True}
    remaining = await client.get(
        "/api/v1/documents", params={"thread_id": "thread_two"}, headers=auth_a
    )
    assert remaining.json()["documents"] == ["thread_two.pdf"]
