"""
WikipediaArticleStore — Persistent SQLite + FTS5 Wikipedia Article Cache.
Provides persistent storage, FTS5 full-text search, and section indexing for Wikipedia articles.
"""
from datetime import datetime
import hashlib
import json
import logging
import re
import sqlite3
import time
import unicodedata
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import settings
from app.persistence.database import get_db_connection, transaction

logger = logging.getLogger("nexia.persistence.article_store")


def _normalize(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


def _hash_content(text: str) -> str:
    """Compute SHA-256 hash prefix for content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class ArticleRecord(BaseModel):
    """Structured persistent Wikipedia article model."""
    article_id: str
    canonical_title: str
    url: str
    language: str = "en"
    summary: str
    content: str = ""
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_timestamp: float = 0.0
    revision_id: Optional[str] = None
    content_hash: str = ""
    source: str = "Wikipedia"
    aliases: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.canonical_title

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        if item == "title":
            return self.canonical_title
        raise KeyError(item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class WikipediaArticleStore:
    """
    Persistent SQLite storage and FTS5 search index for retrieved Wikipedia articles.
    Acts as Level 2 cache in the 3-level retrieval cascade.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or getattr(settings, "article_db_path", "data/wikipedia_articles.db")
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite table, FTS5 virtual table, and sync triggers."""
        with transaction(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wikipedia_articles (
                    article_id TEXT PRIMARY KEY,
                    canonical_title TEXT NOT NULL UNIQUE,
                    url TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    summary TEXT NOT NULL,
                    content TEXT,
                    sections_json TEXT,
                    retrieval_timestamp REAL NOT NULL,
                    revision_id TEXT,
                    content_hash TEXT NOT NULL,
                    source TEXT DEFAULT 'Wikipedia',
                    aliases_json TEXT,
                    metadata_json TEXT
                );
            """)

            # Create FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS wikipedia_articles_fts USING fts5(
                    canonical_title,
                    summary,
                    content,
                    aliases,
                    content='wikipedia_articles',
                    content_rowid='rowid'
                );
            """)

            # SQLite Triggers to keep FTS5 synchronized with the base table
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_articles_ai AFTER INSERT ON wikipedia_articles BEGIN
                    INSERT INTO wikipedia_articles_fts(rowid, canonical_title, summary, content, aliases)
                    VALUES (new.rowid, new.canonical_title, new.summary, coalesce(new.content, ''), coalesce(new.aliases_json, ''));
                END;
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_articles_ad AFTER DELETE ON wikipedia_articles BEGIN
                    INSERT INTO wikipedia_articles_fts(wikipedia_articles_fts, rowid, canonical_title, summary, content, aliases)
                    VALUES ('delete', old.rowid, old.canonical_title, old.summary, coalesce(old.content, ''), coalesce(old.aliases_json, ''));
                END;
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_articles_au AFTER UPDATE ON wikipedia_articles BEGIN
                    INSERT INTO wikipedia_articles_fts(wikipedia_articles_fts, rowid, canonical_title, summary, content, aliases)
                    VALUES ('delete', old.rowid, old.canonical_title, old.summary, coalesce(old.content, ''), coalesce(old.aliases_json, ''));
                    INSERT INTO wikipedia_articles_fts(rowid, canonical_title, summary, content, aliases)
                    VALUES (new.rowid, new.canonical_title, new.summary, coalesce(new.content, ''), coalesce(new.aliases_json, ''));
                END;
            """)

    def save_article(
        self,
        canonical_title: Optional[str] = None,
        url: str = "",
        summary: str = "",
        content: Optional[str] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        revision_id: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        language: str = "en",
        title: Optional[str] = None,
    ) -> str:
        """
        Persist a Wikipedia article into the store.
        Updates existing article if canonical_title already exists.
        """
        raw_title = canonical_title or title or ""
        title_clean = raw_title.strip()
        article_id = _hash_content(title_clean.lower())
        full_content = content or summary
        content_hash = _hash_content(full_content)
        ts = time.time()
        sec_json = json.dumps(sections or [])
        al_json = json.dumps(aliases or [])
        meta_json = json.dumps(metadata or {})

        with transaction(self.db_path) as conn:
            conn.execute("""
                INSERT INTO wikipedia_articles (
                    article_id, canonical_title, url, language, summary,
                    content, sections_json, retrieval_timestamp, revision_id,
                    content_hash, source, aliases_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Wikipedia', ?, ?)
                ON CONFLICT(canonical_title) DO UPDATE SET
                    url=excluded.url,
                    summary=excluded.summary,
                    content=excluded.content,
                    sections_json=excluded.sections_json,
                    retrieval_timestamp=excluded.retrieval_timestamp,
                    revision_id=excluded.revision_id,
                    content_hash=excluded.content_hash,
                    aliases_json=excluded.aliases_json,
                    metadata_json=excluded.metadata_json;
            """, (
                article_id, title_clean, url, language, summary,
                full_content, sec_json, ts, revision_id,
                content_hash, al_json, meta_json
            ))

        logger.debug("[ARTICLE_STORE] Persisted article %r (id=%s)", title_clean, article_id[:8])
        return article_id

    def get_article(self, title_or_alias: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached article by exact canonical title or alias matching.
        """
        if not title_or_alias or not title_or_alias.strip():
            return None

        clean_term = title_or_alias.strip()
        norm_term = _normalize(clean_term)

        with get_db_connection(self.db_path) as conn:
            # 1. Exact match on canonical_title (case-insensitive)
            cursor = conn.execute(
                "SELECT * FROM wikipedia_articles WHERE lower(canonical_title) = lower(?)",
                (clean_term,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)

            # 2. Check all articles for normalized title or alias list match
            cursor = conn.execute("SELECT * FROM wikipedia_articles")
            for r in cursor.fetchall():
                can_norm = _normalize(r["canonical_title"])
                if can_norm == norm_term:
                    return self._row_to_dict(r)
                aliases = json.loads(r["aliases_json"] or "[]")
                if any(_normalize(a) == norm_term for a in aliases):
                    return self._row_to_dict(r)

        return None

    def search_articles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Full-text search across article titles, summaries, sections, and aliases using FTS5.
        """
        if not query or not query.strip():
            return []

        # Sanitize query for FTS5 (strip punctuation and operators)
        clean_words = [w for w in re.findall(r"\w+", query) if len(w) >= 2]
        if not clean_words:
            return []

        fts_query = " OR ".join(f'"{w}"*' for w in clean_words)

        results: List[Dict[str, Any]] = []
        with get_db_connection(self.db_path) as conn:
            try:
                cursor = conn.execute("""
                    SELECT a.* FROM wikipedia_articles a
                    JOIN wikipedia_articles_fts f ON a.rowid = f.rowid
                    WHERE wikipedia_articles_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?;
                """, (fts_query, limit))
                for row in cursor.fetchall():
                    results.append(self._row_to_dict(row))
            except sqlite3.Error as e:
                logger.debug("[ARTICLE_STORE] FTS5 search error (%s): %s", fts_query, e)
                # Fallback to LIKE
                cursor = conn.execute("""
                    SELECT * FROM wikipedia_articles
                    WHERE lower(canonical_title) LIKE lower(?) OR lower(summary) LIKE lower(?)
                    LIMIT ?;
                """, (f"%{clean_words[0]}%", f"%{clean_words[0]}%", limit))
                for row in cursor.fetchall():
                    results.append(self._row_to_dict(row))

        return results

    def save_document(self, doc: Any) -> str:
        """
        Persist a WikipediaDocument instance directly into the store.
        """
        from app.knowledge.document_models import WikipediaDocument
        if not isinstance(doc, WikipediaDocument):
            raise TypeError("Expected WikipediaDocument instance")

        # Convert sections to dict format for storage
        sections_data = [
            {
                "section_id": s.section_id,
                "title": s.title,
                "level": s.level,
                "index": s.index,
                "parent_section_title": s.parent_section_title,
                "content": s.content,
                "passages": [p.model_dump() for p in s.passages],
            }
            for s in doc.sections
        ]

        full_content = "\n\n".join(
            ([doc.lead_summary] if doc.lead_summary else []) +
            [f"== {s.title} ==\n{s.content}" for s in doc.sections if s.content]
        )

        meta = dict(doc.metadata)
        meta["links"] = doc.links[:50]
        meta["categories"] = doc.categories[:20]

        return self.save_article(
            canonical_title=doc.canonical_title,
            url=doc.url,
            summary=doc.lead_summary,
            content=full_content,
            sections=sections_data,
            revision_id=doc.revision_id,
            aliases=doc.aliases,
            metadata=meta,
            language=doc.language,
        )

    def get_document(self, title_or_alias: str) -> Optional[Any]:
        """
        Retrieve a stored article and reconstruct as a WikipediaDocument.
        """
        from app.knowledge.document_models import WikipediaDocument, WikipediaDocumentSection, WikipediaPassage
        art = self.get_article(title_or_alias)
        if not art:
            return None

        raw_sections = art.get("sections", []) or []
        doc_sections: List[WikipediaDocumentSection] = []
        for s in raw_sections:
            raw_passages = s.get("passages", [])
            passages = [
                WikipediaPassage(
                    passage_id=p["passage_id"],
                    canonical_title=p.get("canonical_title", art["canonical_title"]),
                    section_title=p.get("section_title", s.get("title", "")),
                    section_index=p.get("section_index", s.get("index", 0)),
                    passage_index=p.get("passage_index", 0),
                    text=p["text"],
                    token_count_estimate=p.get("token_count_estimate", 0),
                    score=p.get("score", 0.0),
                )
                for p in raw_passages if isinstance(p, dict) and "text" in p
            ]
            doc_sections.append(
                WikipediaDocumentSection(
                    section_id=s.get("section_id", _hash_content(f"{art['canonical_title']}::{s.get('title')}")),
                    title=s.get("title", ""),
                    level=s.get("level", 2),
                    index=s.get("index", len(doc_sections)),
                    parent_section_title=s.get("parent_section_title"),
                    content=s.get("content", ""),
                    passages=passages,
                )
            )

        meta = art.get("metadata", {}) or {}
        return WikipediaDocument(
            document_id=art["article_id"],
            canonical_title=art["canonical_title"],
            page_id=meta.get("page_id"),
            revision_id=art.get("revision_id"),
            url=art["url"],
            language=art.get("language", "en"),
            lead_summary=art["summary"],
            sections=doc_sections,
            links=meta.get("links", []),
            categories=meta.get("categories", []),
            retrieved_at=art.get("retrieval_timestamp", 0.0),
            content_hash=art.get("content_hash", ""),
            aliases=art.get("aliases", []),
            metadata=meta,
        )

    def get_section(self, canonical_title: str, section_title: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific section from a stored article.
        """
        art = self.get_article(canonical_title)
        if not art:
            return None

        norm_sec = _normalize(section_title)
        for sec in art.get("sections", []):
            if _normalize(sec.get("title", "")) == norm_sec or norm_sec in _normalize(sec.get("title", "")):
                return {
                    "canonical_title": art["canonical_title"],
                    "url": art["url"],
                    "section_title": sec.get("title"),
                    "content": sec.get("content", ""),
                }
        return None

    def clear(self) -> None:
        """Clear all stored articles (for test isolation)."""
        with transaction(self.db_path) as conn:
            conn.execute("DELETE FROM wikipedia_articles;")

    def _row_to_dict(self, row: sqlite3.Row) -> ArticleRecord:
        return ArticleRecord(
            article_id=row["article_id"],
            canonical_title=row["canonical_title"],
            url=row["url"],
            language=row["language"],
            summary=row["summary"],
            content=row["content"] or "",
            sections=json.loads(row["sections_json"] or "[]"),
            retrieval_timestamp=float(row["retrieval_timestamp"] or 0.0),
            revision_id=row["revision_id"],
            content_hash=row["content_hash"] or "",
            source=row["source"] or "Wikipedia",
            aliases=json.loads(row["aliases_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )


# Global persistent article store instance
wikipedia_article_store = WikipediaArticleStore()
