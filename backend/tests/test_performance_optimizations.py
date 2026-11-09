"""Comprehensive unit and concurrency test suite for Nexia performance optimizations."""

import asyncio
from datetime import datetime, timezone
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.http_client import get_http_client, close_http_client
from app.main import app
from app.schemas.conversation import ConversationRead, ConversationStatus
from app.services.cache_service import CacheService, InMemoryCache
from app.services.idempotency import IdempotencyService, IdempotencyState
from app.services.rate_limiter import (
    InMemoryTokenBucket,
    RateLimitPolicy,
    RateLimiterService,
)
from app.services.response_controller import (
    InferenceQueueFullError,
    InferenceQueueTimeoutError,
    Priority,
    RequestCancelledError,
    ResponseController,
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_environment():
    """Reset singletons and in-memory caches between tests."""
    yield
    # Reset response controller
    rc = ResponseController.get_instance()
    rc._active_count = 0
    rc._queue.clear()
    rc.active_tasks.clear()
    rc.cancelled_requests.clear()

    # Clear cache
    await CacheService.get_instance().in_memory.clear()
    # Reset idempotency store
    IdempotencyService.get_instance()._records.clear()
    # Close any open clients
    await close_http_client()


# ==============================================================================
# 1. Token Bucket Rate Limiting Tests
# ==============================================================================

class TestTokenBucketRateLimiting:
    """Verifies token consumption, monotonic refill, burst protection, and 429 Retry-After."""

    @pytest.mark.asyncio
    async def test_token_bucket_burst_and_exhaustion(self):
        bucket = InMemoryTokenBucket()
        # Capacity 2, refill every 10s (0.1 tokens/sec), cost 1
        policy = RateLimitPolicy(capacity=2, refill_interval_seconds=10.0, cost=1)

        # 1. First token
        r1 = await bucket.consume("user1", policy)
        assert r1.allowed is True
        assert r1.remaining_tokens == 1.0

        # 2. Second token (burst exhausted)
        r2 = await bucket.consume("user1", policy)
        assert r2.allowed is True
        assert r2.remaining_tokens == 0.0

        # 3. Third token (must be rejected)
        r3 = await bucket.consume("user1", policy)
        assert r3.allowed is False
        assert r3.remaining_tokens == 0.0
        assert r3.retry_after_seconds > 0.0

    @pytest.mark.asyncio
    async def test_token_bucket_monotonic_refill(self):
        bucket = InMemoryTokenBucket()
        # Fast refill: 0.1s per token
        policy = RateLimitPolicy(capacity=1, refill_interval_seconds=0.1, cost=1)

        r1 = await bucket.consume("user2", policy)
        assert r1.allowed is True

        r2 = await bucket.consume("user2", policy)
        assert r2.allowed is False

        # Sleep past refill interval
        await asyncio.sleep(0.15)

        r3 = await bucket.consume("user2", policy)
        assert r3.allowed is True

    @pytest.mark.asyncio
    async def test_rate_limiter_service_independent_endpoints(self):
        service = RateLimiterService.get_instance()
        client_ip = "192.168.1.100"

        # Policy "answer": capacity 2
        p1 = await service.check("answer", client_ip)
        assert p1.allowed is True
        p2 = await service.check("answer", client_ip)
        assert p2.allowed is True
        p3 = await service.check("answer", client_ip)
        assert p3.allowed is False

        # Other policy "conversation_read" has separate bucket and must allow
        c1 = await service.check("conversation_read", client_ip)
        assert c1.allowed is True

    @pytest.mark.asyncio
    async def test_http_429_middleware_behavior(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Send 2 fast requests to /api/v1/retrieval/answer/stream
            # Mock orchestrator stream to return fast
            with patch("app.api.routes.answer.orchestrator.answer_stream") as mock_stream:
                async def _empty_gen(*args, **kwargs):
                    yield "event: start\ndata: {}\n\n"
                mock_stream.side_effect = _empty_gen

                # Req 1
                res1 = await client.post("/api/v1/retrieval/answer/stream", json={"question": "Test 1?"})
                # Req 2
                res2 = await client.post("/api/v1/retrieval/answer/stream", json={"question": "Test 2?"})
                # Req 3 (Exceeds capacity of 2)
                res3 = await client.post("/api/v1/retrieval/answer/stream", json={"question": "Test 3?"})

                assert res3.status_code == 429
                assert "Retry-After" in res3.headers
                data = res3.json()
                assert data["detail"]["error_code"] == "RATE_LIMIT_EXCEEDED"


# ==============================================================================
# 2. Centralized Inference Admission & Bounded Queue Tests
# ==============================================================================

class TestInferenceAdmissionManager:
    """Verifies single-active execution, queue capacity, 503 overflow, 504 timeout, and priority."""

    @pytest.mark.asyncio
    async def test_single_active_slot_admission(self):
        controller = ResponseController(max_active=1, max_queue=3, queue_timeout=10.0)

        started = asyncio.Event()
        release = asyncio.Event()

        async def _long_inference():
            started.set()
            await release.wait()
            return "done"

        # Task 1 occupies the slot
        task1 = asyncio.create_task(
            controller.execute_bounded("req_1", _long_inference)
        )
        await started.wait()
        assert controller.active_generations == 1
        assert controller.queue_depth == 0

        # Clean release
        release.set()
        res = await task1
        assert res == "done"
        assert controller.active_generations == 0

    @pytest.mark.asyncio
    async def test_queue_full_rejection_503(self):
        controller = ResponseController(max_active=1, max_queue=2, queue_timeout=10.0)
        release = asyncio.Event()

        async def _worker():
            await release.wait()

        # 1 active
        t_active = asyncio.create_task(controller.execute_bounded("r_active", _worker))
        await asyncio.sleep(0.01)

        # 2 queued (queue full at 2)
        t_q1 = asyncio.create_task(controller.execute_bounded("r_q1", _worker))
        t_q2 = asyncio.create_task(controller.execute_bounded("r_q2", _worker))
        await asyncio.sleep(0.01)

        assert controller.queue_depth == 2

        # 4th request must be rejected immediately with InferenceQueueFullError
        with pytest.raises(InferenceQueueFullError):
            await controller.acquire_slot("r_overflow", priority=Priority.INTERACTIVE)

        release.set()
        await asyncio.gather(t_active, t_q1, t_q2)

    @pytest.mark.asyncio
    async def test_queue_timeout_504(self):
        # Short queue timeout of 0.1s
        controller = ResponseController(max_active=1, max_queue=3, queue_timeout=0.1)
        release = asyncio.Event()

        async def _blocker():
            await release.wait()

        t_blocker = asyncio.create_task(controller.execute_bounded("blocker", _blocker))
        await asyncio.sleep(0.01)

        # Second request queues and times out
        with pytest.raises(InferenceQueueTimeoutError):
            await controller.acquire_slot("waiting_req", priority=Priority.INTERACTIVE)

        release.set()
        await t_blocker

    @pytest.mark.asyncio
    async def test_priority_scheduling(self):
        controller = ResponseController(max_active=1, max_queue=3, queue_timeout=5.0)
        release = asyncio.Event()

        async def _blocker():
            await release.wait()

        t_blocker = asyncio.create_task(controller.execute_bounded("blocker", _blocker))
        await asyncio.sleep(0.01)

        # Enqueue Background request first
        t_bg = asyncio.create_task(
            controller.acquire_slot("req_bg", priority=Priority.BACKGROUND)
        )
        await asyncio.sleep(0.01)

        # Enqueue Interactive request second
        t_interactive = asyncio.create_task(
            controller.acquire_slot("req_interactive", priority=Priority.INTERACTIVE)
        )
        await asyncio.sleep(0.01)

        # Queue order must have placed interactive BEFORE background!
        assert controller._queue[0].request_id == "req_interactive"
        assert controller._queue[1].request_id == "req_bg"

        release.set()
        await t_blocker
        # Release first slot -> interactive is admitted
        await controller.release_slot("blocker")
        await t_interactive
        # Release second slot -> background is admitted
        await controller.release_slot("req_interactive")
        await t_bg
        await controller.release_slot("req_bg")

    @pytest.mark.asyncio
    async def test_cancellation_queued_request(self):
        controller = ResponseController(max_active=1, max_queue=3, queue_timeout=5.0)
        release = asyncio.Event()

        async def _blocker():
            await release.wait()

        t_block = asyncio.create_task(controller.execute_bounded("blocker", _blocker))
        await asyncio.sleep(0.01)

        t_wait = asyncio.create_task(controller.acquire_slot("target_cancel"))
        await asyncio.sleep(0.01)
        assert controller.queue_depth == 1

        # Cancel queued request
        cancelled = controller.cancel_request("target_cancel")
        assert cancelled is True

        with pytest.raises(RequestCancelledError):
            await t_wait

        assert controller.queue_depth == 0
        release.set()
        await t_block


# ==============================================================================
# 3. Tiered Caching Tests
# ==============================================================================

class TestTieredCaching:
    """Verifies in-memory cache operations, TTL expiration, and prefix invalidation."""

    @pytest.mark.asyncio
    async def test_cache_set_get_and_ttl(self):
        cache = InMemoryCache()
        await cache.set("k1", {"data": "test"}, ttl_seconds=0.1)

        val = await cache.get("k1")
        assert val == {"data": "test"}

        await asyncio.sleep(0.15)
        val_expired = await cache.get("k1")
        assert val_expired is None

    @pytest.mark.asyncio
    async def test_cache_prefix_invalidation(self):
        cache = CacheService.get_instance()
        await cache.set("conv_list:50:0:all", [{"id": "1", "title": "Chat 1"}], ttl_seconds=300)
        await cache.set("conv_list:10:0:active", [{"id": "2", "title": "Chat 2"}], ttl_seconds=300)
        await cache.set("wiki_search:abc", [{"title": "India"}], ttl_seconds=300)

        # Invalidate conversations prefix
        deleted_count = await cache.invalidate_prefix("conv_list:")
        assert deleted_count == 2

        assert await cache.get("conv_list:50:0:all") is None
        assert await cache.get("wiki_search:abc") is not None


# ==============================================================================
# 4. Request Deduplication & Idempotency Tests
# ==============================================================================

class TestIdempotencyService:
    """Verifies duplicate prevention and completed response replay."""

    @pytest.mark.asyncio
    async def test_idempotency_workflow(self):
        svc = IdempotencyService(ttl_seconds=10.0)
        key = "idemp_test_key_1"

        # Request 1 starts
        is_new1, rec1 = await svc.start_request(key)
        assert is_new1 is True

        # Request 2 arrives simultaneously with same key
        is_new2, rec2 = await svc.start_request(key)
        assert is_new2 is False
        assert rec2.state == IdempotencyState.RUNNING

        # Request 1 completes
        expected_response = {"answer": "Sample Grounded Answer"}
        await svc.complete_request(key, expected_response)

        # Request 2 receives completed response
        replayed = await svc.wait_for_result(rec2, timeout=2.0)
        assert replayed == expected_response


# ==============================================================================
# 5. Shared HTTP Client Pool Test
# ==============================================================================

class TestSharedHttpClient:
    """Verifies singleton HTTP client pool lifecycle."""

    @pytest.mark.asyncio
    async def test_http_client_singleton(self):
        c1 = get_http_client()
        c2 = get_http_client()
        assert c1 is c2
        assert not c1.is_closed

        await close_http_client()
        assert c1.is_closed


# ==============================================================================
# 6. Operational Metrics Endpoint Test
# ==============================================================================

class TestMetricsEndpoint:
    """Verifies GET /api/v1/metrics exposes required telemetry sections."""

    @pytest.mark.asyncio
    async def test_get_metrics_payload(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/api/v1/metrics")
            assert res.status_code == 200
            data = res.json()

            assert "inference" in data
            assert "cache" in data
            assert "rate_limiting" in data
            assert "database_pool" in data

            assert data["inference"]["max_active"] >= 1
            assert "queue_depth" in data["inference"]
            assert "hit_ratio" in data["cache"]

