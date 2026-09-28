"""
Memory API endpoints for Nexia Stage 4 long-term persistent memory inspection.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from app.memory.manager import memory_manager
from app.memory.models import Memory, MemoryType

router = APIRouter(prefix="/api/memory", tags=["Memory"])


class StoreMemoryRequest(BaseModel):
    content: str
    user_id: Optional[str] = "default"
    type: Optional[MemoryType] = None
    source: Optional[str] = "user_explicit"


class UpdateMemoryRequest(BaseModel):
    content: str
    user_id: Optional[str] = "default"


@router.get("", response_model=List[Memory])
async def list_memories(
    user_id: str = Query("default"),
    type: Optional[MemoryType] = Query(None),
    all: bool = Query(False, description="Include SUPERSEDED and DELETED records"),
):
    """List memories for a given user."""
    if all:
        return memory_manager.list_all(user_id=user_id)
    return memory_manager.list_active(user_id=user_id, memory_type=type)


@router.post("", response_model=Memory)
async def create_memory(request: StoreMemoryRequest):
    """Explicitly store a memory."""
    mem = memory_manager.store_memory(
        content=request.content,
        user_id=request.user_id or "default",
        memory_type=request.type,
        source=request.source or "user_explicit",
    )
    if not mem:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Memory could not be stored due to policy or empty content.",
        )
    return mem


@router.get("/search", response_model=List[Memory])
async def search_memories(
    query: str = Query(...),
    user_id: str = Query("default"),
    limit: int = Query(5),
):
    """Search relevant active memories."""
    return memory_manager.get_relevant_memories(query=query, user_id=user_id, limit=limit)


@router.get("/{memory_id}", response_model=Memory)
async def get_memory(memory_id: str, user_id: str = Query("default")):
    """Get a specific memory by ID."""
    mem = memory_manager.get_memory(memory_id=memory_id, user_id=user_id)
    if not mem:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")
    return mem


@router.put("/{memory_id}", response_model=Memory)
async def update_memory(memory_id: str, request: UpdateMemoryRequest):
    """Update active memory content."""
    mem = memory_manager.update_memory(
        memory_id=memory_id,
        new_content=request.content,
        user_id=request.user_id or "default",
    )
    if not mem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or not active.",
        )
    return mem


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, user_id: str = Query("default")):
    """Mark a memory as DELETED."""
    deleted = memory_manager.delete_memory(memory_id=memory_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")
    return {"status": "deleted", "id": memory_id}
