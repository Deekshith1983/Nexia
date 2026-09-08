import time
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Message(BaseModel):
    """Structured representation of a single conversation turn."""
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = Field(default_factory=time.time)

class ConversationState(BaseModel):
    """Nexia Core Conversation State tracking canonical conversation history and context."""
    conversation_id: str
    messages: List[Message] = Field(default_factory=list)
    current_topic: Optional[str] = None
    current_entity: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)

class ConversationManager:
    """In-memory manager for creating, retrieving, and updating Nexia conversation states."""

    def __init__(self):
        self._conversations: Dict[str, ConversationState] = {}

    def create_conversation(self, conversation_id: Optional[str] = None) -> ConversationState:
        cid = conversation_id or str(uuid.uuid4())
        state = ConversationState(conversation_id=cid)
        self._conversations[cid] = state
        return state

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        return self._conversations.get(conversation_id)

    def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> ConversationState:
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        return self.create_conversation(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str) -> Message:
        state = self.get_or_create_conversation(conversation_id)
        msg = Message(role=role, content=content)
        state.messages.append(msg)
        return msg

    def update_state(
        self,
        conversation_id: str,
        topic: Optional[str] = None,
        entity: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[ConversationState]:
        state = self.get_conversation(conversation_id)
        if state:
            if topic is not None:
                state.current_topic = topic
            if entity is not None:
                state.current_entity = entity
            if metadata is not None:
                state.metadata.update(metadata)
        return state

# Global Conversation Manager instance
conversation_manager = ConversationManager()
