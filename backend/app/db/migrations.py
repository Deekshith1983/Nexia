"""Database schema initialization and migration helper."""

import logging
from sqlalchemy import text
from app.db.connection import get_engine
from app.db.models import Base

logger = logging.getLogger(__name__)


async def init_db() -> None:
    """Ensure vector extension exists and create all tables and indexes."""
    engine = get_engine()

    async with engine.begin() as conn:
        logger.info("Ensuring 'vector' extension is created...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        logger.info("Creating tables for Nexia models...")
        await conn.run_sync(Base.metadata.create_all)

        logger.info("Creating full-text search and vector indexes if not present...")
        # Full text search GIN index
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_fts "
                "ON nexia_chunks USING gin(to_tsvector('english', chunk_text));"
            )
        )

        # HNSW vector index for cosine distance on chunks
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw "
                    "ON nexia_chunks USING hnsw (embedding vector_cosine_ops);"
                )
            )
        except Exception as e:
            logger.warning("Could not create HNSW index on chunks: %s", e)

        # HNSW vector index for cosine distance on memories
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw "
                    "ON nexia_memories USING hnsw (embedding vector_cosine_ops);"
                )
            )
        except Exception as e:
            logger.warning("Could not create HNSW index on memories: %s", e)

    logger.info("Database initialization complete.")


async def reset_db() -> None:
    """Drop and recreate all tables (useful for isolated testing)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
