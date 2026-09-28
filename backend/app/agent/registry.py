"""
Capability Registry for the Nexia Agent Runtime.

Central registry that maps capability IDs to Capability instances.
The registry does NOT execute capabilities — it only stores and retrieves them.
Execution is the sole responsibility of the Agent Runtime.
"""
from typing import Dict, List, Optional

from app.agent.capability import Capability


class CapabilityRegistry:
    """
    Central store of registered Nexia capabilities.

    Capabilities are registered by ID at application startup.
    The registry is a plain in-memory dict — no database, no persistence.

    Methods:
        register(capability)  — add a capability (overwrites if ID already exists)
        get(capability_id)    — retrieve by ID; returns None if not found
        contains(capability_id) — membership check
        list()                — sorted list of registered capability IDs
    """

    def __init__(self) -> None:
        self._capabilities: Dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        """Register a capability. If the same ID is registered twice, the
        newer registration wins (allows hot-replacement in tests)."""
        self._capabilities[capability.id] = capability

    def get(self, capability_id: str) -> Optional[Capability]:
        """Return the capability for the given ID, or None if not registered."""
        return self._capabilities.get(capability_id)

    def contains(self, capability_id: str) -> bool:
        """Return True if a capability with this ID is registered."""
        return capability_id in self._capabilities

    def list(self) -> List[str]:
        """Return a sorted list of all registered capability IDs."""
        return sorted(self._capabilities.keys())

    def __len__(self) -> int:
        return len(self._capabilities)

    def __repr__(self) -> str:
        return f"CapabilityRegistry(registered={self.list()})"


# Global registry singleton.
# Built-in capabilities are registered by app/agent/__init__.py at import time.
capability_registry = CapabilityRegistry()
