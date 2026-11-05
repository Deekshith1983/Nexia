import logging
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Body, Query, status
from fastapi.responses import StreamingResponse

from app.schemas.answer import AnswerRequest, AnswerResponse
from app.services.answer_orchestrator import AnswerOrchestrator
from app.services.response_controller import ResponseController

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Stage 7 - Grounded Answer Generation"])
orchestrator = AnswerOrchestrator()
resp_controller = ResponseController.get_instance()


@router.post("/retrieval/answer", response_model=AnswerResponse)
@router.post("/retrieve/answer", response_model=AnswerResponse, include_in_schema=False)
async def generate_grounded_answer_post(
    request: AnswerRequest = Body(...),
) -> AnswerResponse:
    """Stage 7: Generate source-grounded answers backed by PostgreSQL vector search and local LLM (POST)."""
    logger.info(">>> [POST /retrieval/answer] Question: '%s'", request.question)
    return await orchestrator.answer(request)


@router.get("/retrieval/answer", response_model=AnswerResponse)
@router.get("/retrieve/answer", response_model=AnswerResponse, include_in_schema=False)
async def generate_grounded_answer_get(
    question: str = Query(..., description="Natural language question to answer"),
    subject_entity: Optional[str] = Query(default=None, description="Optional subject entity hint"),
    object_entity: Optional[str] = Query(default=None, description="Optional object entity hint"),
    predicate_filter: Optional[str] = Query(default=None, description="Optional relationship predicate filter"),
    top_k_per_document: int = Query(default=4, ge=1, le=10, description="Passages per document"),
    min_confidence: float = Query(default=0.60, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    max_evidence_passages: int = Query(default=3, ge=1, le=12, description="Maximum evidence passages sent to LLM"),
    force_refresh: bool = Query(default=False, description="Bypass cache and force re-fetch from Wikipedia"),
) -> AnswerResponse:
    """Stage 7: Generate source-grounded answers backed by PostgreSQL vector search and local LLM (GET)."""
    logger.info(">>> [GET /retrieval/answer] Question: '%s'", question)
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
    return await orchestrator.answer(request)


@router.post("/retrieval/answer/stream")
@router.post("/retrieve/answer/stream", include_in_schema=False)
async def generate_grounded_answer_stream_post(
    request: AnswerRequest = Body(...),
) -> StreamingResponse:
    """Stream grounded answer tokens progressively via Server-Sent Events (SSE) (POST)."""
    logger.info(">>> [POST /retrieval/answer/stream] Question: '%s'", request.question)
    return StreamingResponse(
        orchestrator.answer_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/retrieval/answer/stream")
@router.get("/retrieve/answer/stream", include_in_schema=False)
async def generate_grounded_answer_stream_get(
    question: str = Query(..., description="Natural language question to answer"),
    conversation_id: Optional[UUID] = Query(default=None, description="Optional conversation session UUID"),
    force_refresh: bool = Query(default=False),
) -> StreamingResponse:
    """Stream grounded answer tokens progressively via Server-Sent Events (SSE) (GET)."""
    logger.info(">>> [GET /retrieval/answer/stream] Question: '%s'", question)
    request = AnswerRequest(
        question=question,
        conversation_id=conversation_id,
        force_refresh=force_refresh,
    )
    return StreamingResponse(
        orchestrator.answer_stream(request),
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
