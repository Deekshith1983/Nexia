import logging
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Body, Depends, HTTPException, Header, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.middleware.rate_limit import rate_limit
from app.schemas.answer import (
    MAX_EVIDENCE_PASSAGES,
    MIN_CONFIDENCE,
    TOP_K_PER_DOCUMENT,
    AnswerRequest,
    AnswerResponse,
)
from app.services.answer_orchestrator import AnswerOrchestrator
from app.services.idempotency import IdempotencyService
from app.services.response_controller import (
    InferenceQueueFullError,
    InferenceQueueTimeoutError,
    RequestCancelledError,
    ResponseController,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Stage 7 - Grounded Answer Generation"])
orchestrator = AnswerOrchestrator()
resp_controller = ResponseController.get_instance()
idempotency_svc = IdempotencyService.get_instance()


@router.post(
    "/retrieval/answer",
    response_model=AnswerResponse,
    dependencies=[Depends(rate_limit("answer"))],
)
@router.post(
    "/retrieve/answer",
    response_model=AnswerResponse,
    include_in_schema=False,
    dependencies=[Depends(rate_limit("answer"))],
)
async def generate_grounded_answer_post(
    raw_request: Request,
    request: AnswerRequest = Body(...),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
) -> AnswerResponse:
    """Stage 7: Generate source-grounded answers backed by PostgreSQL vector search and local LLM (POST)."""
    key = idempotency_key or raw_request.headers.get("X-Request-ID")
    req_id = getattr(raw_request.state, "request_id", None) or key

    if key:
        is_new, record = await idempotency_svc.start_request(key)
        if not is_new and record:
            cached_data = await idempotency_svc.wait_for_result(record)
            if cached_data:
                logger.info("Serving idempotent cached answer response for key: %s", key)
                return AnswerResponse.model_validate(cached_data)

    logger.info(">>> [POST /retrieval/answer] Question: '%s'", request.question)
    try:
        result = await orchestrator.answer(request, request_id=req_id)
        if key:
            await idempotency_svc.complete_request(key, result.model_dump())
        return result
    except InferenceQueueFullError as exc:
        if key:
            await idempotency_svc.fail_request(key, exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except InferenceQueueTimeoutError as exc:
        if key:
            await idempotency_svc.fail_request(key, exc)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
    except RequestCancelledError as exc:
        if key:
            await idempotency_svc.cancel_request(key)
        raise HTTPException(status_code=499, detail="Generation was cancelled")
    except Exception as exc:
        if key:
            await idempotency_svc.fail_request(key, exc)
        raise


@router.get(
    "/retrieval/answer",
    response_model=AnswerResponse,
    include_in_schema=False,
    dependencies=[Depends(rate_limit("answer"))],
)
@router.get(
    "/retrieve/answer",
    response_model=AnswerResponse,
    include_in_schema=False,
    dependencies=[Depends(rate_limit("answer"))],
)
async def generate_grounded_answer_get(
    raw_request: Request,
    question: str = Query(..., description="Natural language question to answer"),
    subject_entity: Optional[str] = Query(default=None, description="Optional subject entity hint"),
    object_entity: Optional[str] = Query(default=None, description="Optional object entity hint"),
    predicate_filter: Optional[str] = Query(default=None, description="Optional relationship predicate filter"),
    top_k_per_document: int = Query(default=TOP_K_PER_DOCUMENT, ge=1, le=10, description="Passages per document"),
    min_confidence: float = Query(default=MIN_CONFIDENCE, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    max_evidence_passages: int = Query(default=MAX_EVIDENCE_PASSAGES, ge=1, le=12, description="Maximum evidence passages sent to LLM"),
    force_refresh: bool = Query(default=False, description="Bypass cache and force re-fetch from Wikipedia"),
) -> AnswerResponse:
    """Stage 7: Generate source-grounded answers backed by PostgreSQL vector search and local LLM (GET)."""
    logger.info(">>> [GET /retrieval/answer] Question: '%s'", question)
    req_id = getattr(raw_request.state, "request_id", None)
    request = AnswerRequest(
        question=question,
        subject_entity=subject_entity,
        object_entity=object_entity,
        predicate_filter=predicate_filter,
        top_k_per_document=top_k_per_document,
        min_confidence=min_confidence,
        max_evidence_passages=max_evidence_passages,
        force_refresh=force_refresh,
    )
    try:
        return await orchestrator.answer(request, request_id=req_id)
    except InferenceQueueFullError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except InferenceQueueTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
    except RequestCancelledError:
        raise HTTPException(status_code=499, detail="Generation was cancelled")


@router.post(
    "/retrieval/answer/stream",
    dependencies=[Depends(rate_limit("answer"))],
)
@router.post(
    "/retrieve/answer/stream",
    include_in_schema=False,
    dependencies=[Depends(rate_limit("answer"))],
)
async def generate_grounded_answer_stream_post(
    raw_request: Request,
    request: AnswerRequest = Body(...),
) -> StreamingResponse:
    """Stream grounded answer tokens progressively via Server-Sent Events (SSE) (POST)."""
    req_id = getattr(raw_request.state, "request_id", None)
    logger.info(">>> [POST /retrieval/answer/stream] Question: '%s'", request.question)
    return StreamingResponse(
        orchestrator.answer_stream(request, request_id=req_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/retrieval/answer/stream",
    include_in_schema=False,
    dependencies=[Depends(rate_limit("answer"))],
)
@router.get(
    "/retrieve/answer/stream",
    include_in_schema=False,
    dependencies=[Depends(rate_limit("answer"))],
)
async def generate_grounded_answer_stream_get(
    raw_request: Request,
    question: str = Query(..., description="Natural language question to answer"),
    conversation_id: Optional[UUID] = Query(default=None, description="Optional conversation session UUID"),
    force_refresh: bool = Query(default=False),
) -> StreamingResponse:
    """Stream grounded answer tokens progressively via Server-Sent Events (SSE) (GET)."""
    req_id = getattr(raw_request.state, "request_id", None)
    logger.info(">>> [GET /retrieval/answer/stream] Question: '%s'", question)
    request = AnswerRequest(
        question=question,
        conversation_id=conversation_id,
        force_refresh=force_refresh,
    )
    return StreamingResponse(
        orchestrator.answer_stream(request, request_id=req_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/retrieval/cancel/{request_id}", status_code=status.HTTP_200_OK)
async def cancel_generation(request_id: str):
    """Cancel an in-flight LLM answer generation request."""
    cancelled = resp_controller.cancel_request(request_id)
    return {"request_id": request_id, "cancelled": cancelled}
