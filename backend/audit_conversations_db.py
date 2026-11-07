import asyncio
from app.db.connection import get_session_factory, get_engine
from app.db.models import NexiaConversation, NexiaConversationMessage
from sqlalchemy import select

async def inspect():
    sf = get_session_factory()
    try:
        async with sf() as session:
            res = await session.execute(
                select(NexiaConversation).order_by(NexiaConversation.created_at.asc())
            )
            convs = res.scalars().all()
            print(f"Total conversations: {len(convs)}")
            for c in convs:
                msg_res = await session.execute(
                    select(NexiaConversationMessage)
                    .where(NexiaConversationMessage.conversation_id == c.id)
                    .order_by(NexiaConversationMessage.created_at.asc())
                )
                msgs = msg_res.scalars().all()
                msg_previews = [f"[{m.role}]: {m.content[:40]}..." for m in msgs]
                print(f"ID: {c.id} | Title: '{c.title}' | MsgCount: {len(msgs)} | Created: {c.created_at} | Updated: {c.updated_at}")
                for p in msg_previews:
                    print(f"    {p}")
    finally:
        engine = get_engine()
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(inspect())

