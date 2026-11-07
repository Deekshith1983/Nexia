"""Safe cleanup script for test-generated duplicate conversations in PostgreSQL.

Rules:
1. Preserve user conversations:
   - 'History Discussion' (7c239df1-ddf8-49ce-bc66-e219e8d79353)
   - 'string' (1a631b26-8f9c-48b6-8c98-c6c9bc7ba51f)
2. For test-fixture titles ('Reign of Ashoka', 'My Custom Research', 'Shivaji The Warrior King', 'Shivaji Maharaj Reign'):
   - Keep exactly ONE canonical instance (the latest valid one).
   - Delete older identical test-run duplicates.
3. Delete 0-message empty test chats ('Test Chat', empty 'New Chat' test residue).
"""

import asyncio
from app.db.connection import get_session_factory, get_engine
from app.db.models import NexiaConversation, NexiaConversationMessage
from sqlalchemy import select, delete

PRESERVE_CONVERSATION_IDS = {
    "7c239df1-ddf8-49ce-bc66-e219e8d79353",
    "1a631b26-8f9c-48b6-8c98-c6c9bc7ba51f",
}

TEST_FIXTURE_TITLES = {
    "Reign of Ashoka",
    "My Custom Research",
    "Shivaji The Warrior King",
    "Shivaji Maharaj Reign",
}

async def cleanup():
    sf = get_session_factory()
    try:
        async with sf() as session:
            res = await session.execute(
                select(NexiaConversation).order_by(NexiaConversation.created_at.desc())
            )
            all_convs = res.scalars().all()
            print(f"Total conversations before cleanup: {len(all_convs)}")

            kept_titles = set()
            ids_to_delete = []

            for c in all_convs:
                str_id = str(c.id)
                if str_id in PRESERVE_CONVERSATION_IDS:
                    print(f"PRESERVING user conversation: {c.title} ({c.id})")
                    continue

                # Check message count
                msg_res = await session.execute(
                    select(NexiaConversationMessage).where(NexiaConversationMessage.conversation_id == c.id)
                )
                msgs = msg_res.scalars().all()
                msg_count = len(msgs)

                # 0-message test chat residue
                if msg_count == 0:
                    print(f"MARKING FOR DELETION (0 messages): {c.title} ({c.id})")
                    ids_to_delete.append(c.id)
                    continue

                if c.title in TEST_FIXTURE_TITLES:
                    if c.title not in kept_titles:
                        print(f"KEEPING 1 canonical instance of '{c.title}': ({c.id})")
                        kept_titles.add(c.title)
                    else:
                        print(f"MARKING FOR DELETION duplicate test fixture '{c.title}': ({c.id})")
                        ids_to_delete.append(c.id)
                    continue

                # Any other conversation with messages: preserve
                print(f"PRESERVING conversation: {c.title} ({c.id})")

            print(f"\nTotal to delete: {len(ids_to_delete)}")
            for del_id in ids_to_delete:
                stmt = select(NexiaConversation).where(NexiaConversation.id == del_id)
                r = await session.execute(stmt)
                conv_to_del = r.scalar_one_or_none()
                if conv_to_del:
                    await session.delete(conv_to_del)
            
            await session.commit()
            print("Cleanup committed successfully.")

            # Verify remaining count
            res2 = await session.execute(
                select(NexiaConversation).order_by(NexiaConversation.created_at.desc())
            )
            remaining = res2.scalars().all()
            print(f"Total conversations after cleanup: {len(remaining)}")
            for rem in remaining:
                print(f"  - {rem.title} ({rem.id}) | Updated: {rem.updated_at}")
    finally:
        engine = get_engine()
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(cleanup())

