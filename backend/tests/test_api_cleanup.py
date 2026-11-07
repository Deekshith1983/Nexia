"""Tests verifying FastAPI endpoint cleanup and API surface minimization."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.conversation import (
    ChatMessage,
    ConversationRead,
    ConversationStatus,
    ConversationWithHistory,
    MessageRole,
)

client = TestClient(app)


# ============================================================================
# 1. Verification of Removed Endpoints (Must Return 404 Not Found)
# ============================================================================

@pytest.mark.parametrize(
    "method,path",
    [
        # Standalone Retrieval Operations (API v1)
        ("GET", "/api/v1/retrieval/search"),
        ("GET", "/api/v1/retrieval/document"),
        ("POST", "/api/v1/retrieval/passages"),
        ("GET", "/api/v1/retrieval/passages"),
        ("POST", "/api/v1/retrieval/multi-document"),
        ("GET", "/api/v1/retrieval/multi-document"),
        # Standalone Stage 5 Relationships (API v1)
        ("POST", "/api/v1/retrieval/relationships"),
        ("GET", "/api/v1/retrieval/relationships"),
        # Standalone Stage 6 Evidence Reasoning (API v1)
        ("POST", "/api/v1/retrieval/evidence"),
        ("GET", "/api/v1/retrieval/evidence"),
        # Standalone Unprefixed Routes
        ("GET", "/retrieval/search"),
        ("GET", "/retrieval/document"),
        ("POST", "/retrieval/passages"),
        ("GET", "/retrieval/passages"),
        ("POST", "/retrieval/multi-document"),
        ("GET", "/retrieval/multi-document"),
        ("POST", "/retrieval/relationships"),
        ("GET", "/retrieval/relationships"),
        ("POST", "/retrieval/evidence"),
        ("GET", "/retrieval/evidence"),
        # Duplicate Unprefixed Route Registrations
        ("GET", "/conversations"),
        ("POST", "/conversations"),
        ("POST", "/retrieval/answer"),
        ("POST", "/retrieval/answer/stream"),
        ("POST", "/retrieval/cancel/dummy-req"),
    ],
)
def test_removed_endpoints_return_404(method: str, path: str):
    """Ensure obsolete and duplicate endpoints are no longer registered and return 404."""
    if method == "GET":
        resp = client.get(path)
    elif method == "POST":
        resp = client.post(path, json={"dummy": "data"})
    assert resp.status_code == 404, f"Expected 404 for {method} {path}, got {resp.status_code}"


# ============================================================================
# 2. Verification of Approved Retained Endpoints
# ============================================================================

def test_health_endpoints():
    """Verify health endpoints respond with 200 OK at both /health and /api/v1/health."""
    r_root = client.get("/health")
    assert r_root.status_code == 200
    assert r_root.json() == {"status": "ok"}

    r_v1 = client.get("/api/v1/health")
    assert r_v1.status_code == 200
    assert r_v1.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_conversation_lifecycle_endpoints():
    """Verify conversation CRUD endpoints are mounted under /api/v1/conversations."""
    conv_id = uuid4()
    mock_conv = ConversationRead(
        id=conv_id,
        title="Test Chat",
        status=ConversationStatus.ACTIVE,
        metadata={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        message_count=0,
    )
    mock_history = ConversationWithHistory(
        id=conv_id,
        title="Test Chat",
        status=ConversationStatus.ACTIVE,
        metadata={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        message_count=0,
        messages=[],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        from app.api.routes import conversations as conv_routes
        with patch.object(conv_routes.conv_manager, "create_conversation", AsyncMock(return_value=mock_conv)), \
             patch.object(conv_routes.conv_manager, "list_conversations", AsyncMock(return_value=[mock_conv])), \
             patch.object(conv_routes.conv_manager, "get_conversation_with_history", AsyncMock(return_value=mock_history)), \
             patch.object(conv_routes.conv_manager, "update_conversation", AsyncMock(return_value=mock_conv)), \
             patch.object(conv_routes.conv_manager, "get_messages", AsyncMock(return_value=[])), \
             patch.object(conv_routes.conv_manager, "delete_conversation", AsyncMock(return_value=True)):

            # Create conversation
            create_resp = await ac.post("/api/v1/conversations", json={"title": "Test Chat"})
            assert create_resp.status_code == 201
            assert create_resp.json()["id"] == str(conv_id)

            # List conversations
            list_resp = await ac.get("/api/v1/conversations")
            assert list_resp.status_code == 200
            assert len(list_resp.json()) == 1

            # Get single conversation
            get_resp = await ac.get(f"/api/v1/conversations/{conv_id}")
            assert get_resp.status_code == 200
            assert get_resp.json()["id"] == str(conv_id)

            # Update conversation
            update_resp = await ac.patch(f"/api/v1/conversations/{conv_id}", json={"title": "Renamed Chat"})
            assert update_resp.status_code == 200

            # Get conversation messages
            msgs_resp = await ac.get(f"/api/v1/conversations/{conv_id}/messages")
            assert msgs_resp.status_code == 200
            assert isinstance(msgs_resp.json(), list)

            # Delete conversation
            del_resp = await ac.delete(f"/api/v1/conversations/{conv_id}")
            assert del_resp.status_code == 204


def test_cancel_generation_endpoint():
    """Verify POST /api/v1/retrieval/cancel/{request_id} cancels active generation."""
    resp = client.post("/api/v1/retrieval/cancel/test-request-uuid-1234")
    assert resp.status_code == 200
    data = resp.json()
    assert "cancelled" in data
    assert data["request_id"] == "test-request-uuid-1234"


def test_root_index_endpoint():
    """Verify root GET / returns metadata with only approved endpoints."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "endpoints" in data
    endpoints = data["endpoints"]
    assert endpoints["health"] == "/api/v1/health"
    assert endpoints["conversations"] == "/api/v1/conversations"
    assert endpoints["answer"] == "/api/v1/retrieval/answer"
    assert endpoints["answer_stream"] == "/api/v1/retrieval/answer/stream"
    assert "search" not in endpoints
    assert "document" not in endpoints
    assert "passages" not in endpoints
    assert "multi_document" not in endpoints
    assert "relationships" not in endpoints
    assert "evidence" not in endpoints


# ============================================================================
# 3. Verification of OpenAPI / Swagger Schema
# ============================================================================

def test_openapi_schema_contains_only_approved_routes():
    """Verify the generated OpenAPI schema only exposes the minimal frontend API surface."""
    openapi = app.openapi()
    paths = set(openapi["paths"].keys())

    # Approved paths that MUST be in Swagger
    approved_paths = {
        "/",
        "/health",
        "/api/v1/health",
        "/api/v1/conversations",
        "/api/v1/conversations/{conversation_id}",
        "/api/v1/conversations/{conversation_id}/messages",
        "/api/v1/retrieval/answer",
        "/api/v1/retrieval/answer/stream",
        "/api/v1/retrieval/cancel/{request_id}",
    }

    # Verify all approved paths are present
    for p in approved_paths:
        assert p in paths, f"Approved path {p} missing from OpenAPI schema!"

    # Verify obsolete paths are NOT in Swagger
    forbidden_substrings = [
        "/search",
        "/document",
        "/passages",
        "/multi-document",
        "/relationships",
        "/evidence",
    ]
    for path in paths:
        for forbidden in forbidden_substrings:
            assert forbidden not in path, f"Forbidden route {path} found in OpenAPI schema!"
