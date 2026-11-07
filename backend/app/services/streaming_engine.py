"""Streaming Engine: Formats and delivers Server-Sent Events (SSE) for grounded answer generation."""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID

from app.schemas.answer import SourceReference, TimingBreakdown

logger = logging.getLogger("nexia.streaming_engine")


class SSEEvent:
    """Helper to format SSE chunks according to W3C EventSource specifications."""

    @staticmethod
    def format(event: str, data: Dict[str, Any]) -> str:
        payload = json.dumps(data, default=str)
        return f"event: {event}\ndata: {payload}\n\n"


class StreamingEngine:
    """Manages SSE stream formatting and safe verified delivery."""

    @classmethod
    def emit_start(
        cls,
        request_id: str,
        conversation_id: Optional[UUID],
        model: str,
    ) -> str:
        return SSEEvent.format(
            "start",
            {
                "request_id": request_id,
                "conversation_id": str(conversation_id) if conversation_id else None,
                "model": model,
                "status": "STARTED",
            },
        )

    @classmethod
    def emit_progress(
        cls,
        request_id: str,
        stage: str,
        message: str,
    ) -> str:
        return SSEEvent.format(
            "progress",
            {
                "request_id": request_id,
                "stage": stage,
                "message": message,
            },
        )

    @classmethod
    def emit_token(
        cls,
        request_id: str,
        token: str,
    ) -> str:
        return SSEEvent.format(
            "token",
            {
                "request_id": request_id,
                "token": token,
            },
        )

    @classmethod
    def emit_citation(
        cls,
        request_id: str,
        citations: List[Dict[str, Any]],
    ) -> str:
        return SSEEvent.format(
            "citation",
            {
                "request_id": request_id,
                "citations": citations,
            },
        )

    @classmethod
    def emit_complete(
        cls,
        request_id: str,
        conversation_id: Optional[UUID],
        message_id: Optional[UUID],
        answer_text: str,
        answer_status: str,
        source_references: List[Dict[str, Any]],
        diagnostics: Dict[str, Any],
        timings: Optional[Dict[str, Any]] = None,
    ) -> str:
        return SSEEvent.format(
            "complete",
            {
                "request_id": request_id,
                "conversation_id": str(conversation_id) if conversation_id else None,
                "message_id": str(message_id) if message_id else None,
                "answer_text": answer_text,
                "answer_status": answer_status,
                "source_references": source_references,
                "diagnostics": diagnostics,
                "timings": timings or {},
            },
        )

    @classmethod
    def emit_title(
        cls,
        request_id: str,
        conversation_id: Optional[UUID],
        title: str,
    ) -> str:
        return SSEEvent.format(
            "title",
            {
                "request_id": request_id,
                "conversation_id": str(conversation_id) if conversation_id else None,
                "title": title,
            },
        )

    @classmethod
    def emit_error(
        cls,
        request_id: str,
        error_code: str,
        message: str,
    ) -> str:
        return SSEEvent.format(
            "error",
            {
                "request_id": request_id,
                "error_code": error_code,
                "message": message,
            },
        )

    @classmethod
    def emit_cancelled(
        cls,
        request_id: str,
        reason: str = "Client requested cancellation",
    ) -> str:
        return SSEEvent.format(
            "cancelled",
            {
                "request_id": request_id,
                "reason": reason,
            },
        )

