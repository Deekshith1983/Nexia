"""
Nexia Agent package.

Exports the agent_runtime singleton and registers built-in capabilities.

Import order matters:
    1. Registry singleton created (registry.py)
    2. Capability classes imported (capabilities/*.py)
    3. Built-in capabilities registered into the singleton
    4. Runtime singleton created (runtime.py) — imports registry
    5. Public symbols exported
"""
from app.agent.registry import capability_registry, CapabilityRegistry
from app.agent.capability import Capability, CapabilityRequest, CapabilityResult, CapabilityStatus
from app.agent.execution_context import ExecutionContext, AgentPlan

# Register built-in capabilities at package import time.
# Order of registration determines list() order (sorted by ID in the registry).
from app.agent.capabilities.conversation import ConversationCapability
from app.agent.capabilities.wikipedia import WikipediaKnowledgeCapability
from app.agent.capabilities.memory import MemoryCapability
from app.agent.capabilities.session_op import ResearchSessionCapability

capability_registry.register(ConversationCapability())
capability_registry.register(WikipediaKnowledgeCapability())
capability_registry.register(MemoryCapability())
capability_registry.register(ResearchSessionCapability())

from app.agent.runtime import agent_runtime, NexiaAgentRuntime  # noqa: E402

__all__ = [
    # Runtime
    "agent_runtime",
    "NexiaAgentRuntime",
    # Registry
    "capability_registry",
    "CapabilityRegistry",
    # Abstractions
    "Capability",
    "CapabilityRequest",
    "CapabilityResult",
    "CapabilityStatus",
    # Context & Plan
    "ExecutionContext",
    "AgentPlan",
]
