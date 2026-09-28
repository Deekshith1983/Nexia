"""
MemoryCapability — Handles explicit user memory operations (remember, forget, recall).
"""
import logging
import time
from typing import List

from app.agent.capability import Capability, CapabilityRequest, CapabilityResult, CapabilityStatus
from app.core.conversation_manager import conversation_manager
from app.core.source_attribution import SourceMetadata
from app.memory.commands import MemoryCommand, MemoryCommandType, memory_command_parser
from app.memory.manager import memory_manager

logger = logging.getLogger("nexia.agent.memory")

CAPABILITY_ID = "memory_operation"


class MemoryCapability(Capability):
    """
    Handles explicit user memory instructions (REMEMBER, FORGET, RECALL).
    Does NOT trigger Wikipedia retrieval or unnecessary LLM generation calls.
    """

    @property
    def id(self) -> str:
        return CAPABILITY_ID

    @property
    def description(self) -> str:
        return (
            "Handles user-controlled persistent memory operations "
            "(remember, forget, recall) without external search."
        )

    async def execute(self, request: CapabilityRequest) -> CapabilityResult:
        ctx = request.context
        cid = ctx.conversation_id
        user_id = ctx.metadata.get("user_id", "default")
        t_start = time.monotonic()

        mem_cmd: MemoryCommand = request.input.get("mem_cmd")
        if not mem_cmd:
            mem_cmd = memory_command_parser.parse(ctx.original_query)

        # Record user message in conversation history
        conversation_manager.add_message(cid, "user", ctx.original_query)

        response_text = ""
        if not mem_cmd:
            response_text = "I wasn't able to understand the memory instruction."
        elif mem_cmd.command_type == MemoryCommandType.REMEMBER:
            saved = memory_manager.store_memory(
                content=mem_cmd.target_content,
                user_id=user_id,
                memory_type=mem_cmd.target_type,
                source="user_explicit",
            )
            if saved:
                response_text = f"I'll remember that: {saved.content}"
            else:
                response_text = "I couldn't save that memory."

        elif mem_cmd.command_type == MemoryCommandType.FORGET:
            deleted_count = memory_manager.forget_memory(
                query_or_pattern=mem_cmd.target_content,
                user_id=user_id,
            )
            if deleted_count > 0:
                response_text = f"I've forgotten what I remembered about: {mem_cmd.target_content}"
            else:
                response_text = f"I couldn't find any active memories matching '{mem_cmd.target_content}'."

        elif mem_cmd.command_type == MemoryCommandType.RECALL:
            memories = memory_manager.recall_memories(
                query=mem_cmd.target_content,
                user_id=user_id,
            )
            if memories:
                lines = [f"- {m.content}" for m in memories]
                response_text = "Here is what I remember:\n" + "\n".join(lines)
            else:
                response_text = f"I don't have any saved memories about {mem_cmd.target_content}."

        sources: List[SourceMetadata] = []

        async def stream_gen():
            conversation_manager.add_message(cid, "assistant", response_text)
            total_ms = int((time.monotonic() - t_start) * 1000)
            logger.debug(
                "[MEM CAP] cid=%s user_id=%s cmd=%s total_ms=%d",
                cid[:8], user_id, mem_cmd.command_type if mem_cmd else "NONE", total_ms
            )
            yield response_text

        return CapabilityResult(
            status=CapabilityStatus.SUCCESS,
            stream=stream_gen(),
            sources=sources,
            metadata={
                "capability": CAPABILITY_ID,
                "conversation_id": cid,
                "user_id": user_id,
                "command_type": mem_cmd.command_type.value if mem_cmd else None,
            },
        )
