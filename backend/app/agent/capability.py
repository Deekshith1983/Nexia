"""
Core capability abstractions for the Nexia Agent Runtime.

Defines the Capability interface, CapabilityRequest, CapabilityResult,
and CapabilityStatus — the contract every capability must satisfy.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from app.core.source_attribution import SourceMetadata

if TYPE_CHECKING:
    from app.agent.execution_context import ExecutionContext


class CapabilityStatus(str, Enum):
    """Enumerated outcome states for a capability execution."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    CANCELLED = "CANCELLED"


@dataclass
class CapabilityResult:
    """
    Structured result of a single capability execution.

    Fields:
        status    — enumerated outcome
        output    — optional pre-generated text (None for streaming capabilities)
        stream    — async generator for streaming responses; None for non-streaming
        sources   — Wikipedia / external source metadata (may be empty list)
        error     — human-readable error message if status is not SUCCESS
        metadata  — lightweight observability payload (timings, page title, etc.)
    """
    status: CapabilityStatus
    output: Optional[str] = None
    stream: Optional[AsyncGenerator[str, None]] = None
    sources: List[SourceMetadata] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityRequest:
    """
    Structured invocation request sent from the Agent Runtime to a Capability.

    Fields:
        capability_id — registry key identifying the target capability
        input         — typed dict with capability-specific execution parameters
                        (includes pre-computed `qu`, `resolution`, `plan` objects)
        context       — ExecutionContext for the current agent run
        metadata      — additional call-site metadata (e.g. run_id)
    """
    capability_id: str
    input: Dict[str, Any]
    context: "ExecutionContext"
    metadata: Dict[str, Any] = field(default_factory=dict)


class Capability(ABC):
    """
    Abstract base class for all Nexia capabilities.

    A capability represents a meaningful, user-facing ability that Nexia
    can invoke (e.g. answer a factual question, hold a conversation).

    Internal subsystems (QU, context resolution, prompt construction, etc.)
    are NOT capabilities — they are implementation details owned by modules
    and assembled by the Agent Runtime or the capability itself.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique registry key for this capability."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this capability does."""
        ...

    @abstractmethod
    async def execute(self, request: CapabilityRequest) -> CapabilityResult:
        """
        Execute the capability and return a structured result.

        The result may carry a streaming async generator in `result.stream`.
        The caller is responsible for consuming (or discarding) the generator.

        Must not raise — all errors should be returned as CapabilityResult
        with status FAILED or CANCELLED as appropriate.
        """
        ...
