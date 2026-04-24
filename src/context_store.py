"""
RAG Context Store — ChromaDB-backed retrieval for agent chat.

Replaces the old "prompt-stuffing" approach by indexing meeting data
into 3 ChromaDB collections and exposing a single `search()` interface
for the agent to retrieve only the most relevant context per turn.

Collections:
  transcript_chunks  — 3-sentence sliding-window chunks of each meeting transcript
  graph_events       — Event summaries from TemporalGraphMemory
  meeting_directory  — One doc per meeting (title + participants + date)
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Adapter: let ChromaDB use the *already-loaded* MiniLM model from
# TextAnalyzer instead of downloading its own embedding model.
# ──────────────────────────────────────────────────────────────────────

class _MiniLMEmbeddingFunction(chromadb.EmbeddingFunction):
    """Bridges TextAnalyzer.get_embedding() → ChromaDB embedding fn."""

    def __init__(self, text_analyzer):
        self._ta = text_analyzer

    def __call__(self, input: list[str]) -> list[list[float]]:
        results = []
        for text in input:
            emb = self._ta.get_embedding(text)
            if emb is not None:
                vec = emb.tolist() if isinstance(emb, np.ndarray) else emb
            else:
                vec = [0.0] * 384  # MiniLM dimension; zero-vec fallback
            results.append(vec)
        return results


class ContextStore:
    """
    Thin wrapper around a persistent ChromaDB instance.
    All collections share the same MiniLM embedding function so we
    never load a second model.
    """

    CHROMA_DIR = "chroma_db"

    # Collection names
    COL_TRANSCRIPT  = "transcript_chunks"
    COL_EVENTS      = "graph_events"
    COL_MEETINGS    = "meeting_directory"

    def __init__(self, storage_root: str = ".", text_analyzer=None):
        persist_dir = os.path.join(storage_root, self.CHROMA_DIR)
        self._client = chromadb.PersistentClient(path=persist_dir)

        if text_analyzer is not None:
            self._emb_fn = _MiniLMEmbeddingFunction(text_analyzer)
        else:
            self._emb_fn = None  # ChromaDB default (not recommended)
            logger.warning("ContextStore created without TextAnalyzer — using ChromaDB default embeddings")

        self._transcript = self._client.get_or_create_collection(
            name=self.COL_TRANSCRIPT,
            embedding_function=self._emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._events = self._client.get_or_create_collection(
            name=self.COL_EVENTS,
            embedding_function=self._emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._meetings = self._client.get_or_create_collection(
            name=self.COL_MEETINGS,
            embedding_function=self._emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ContextStore ready — transcript:{self._transcript.count()}, "
            f"events:{self._events.count()}, meetings:{self._meetings.count()}"
        )

    # ================================================================
    # Indexing
    # ================================================================

    def index_meeting(self, job: Dict[str, Any], temporal_memory=None) -> Dict[str, int]:
        """
        Index a completed meeting into all 3 collections.
        Should be called once after processing finishes.

        Returns counts: {"transcript_chunks": N, "events": N, "meeting_dir": 1}
        """
        meeting_id = job.get("meeting_id") or job.get("job_id", "unknown")
        counts = {"transcript_chunks": 0, "events": 0, "meeting_dir": 0}

        # ── 1. Transcript chunks ──────────────────────────────────────
        transcript = job.get("transcript", "")
        if transcript:
            chunks = self._chunk_transcript(transcript, window=3)
            if chunks:
                ids = [f"{meeting_id}_tc_{i}" for i in range(len(chunks))]
                docs = [c["text"] for c in chunks]
                metadatas = [
                    {
                        "meeting_id": meeting_id,
                        "speaker": c.get("speaker", ""),
                        "start_time": c.get("start_time", ""),
                        "chunk_index": i,
                    }
                    for i, c in enumerate(chunks)
                ]
                self._transcript.upsert(ids=ids, documents=docs, metadatas=metadatas)
                counts["transcript_chunks"] = len(chunks)

        # ── 2. Graph events ───────────────────────────────────────────
        if temporal_memory and meeting_id:
            event_ids = temporal_memory.events_by_meeting.get(meeting_id, [])
            ev_docs, ev_ids, ev_metas = [], [], []
            for eid in event_ids:
                ev = temporal_memory.events.get(eid)
                if not ev:
                    continue
                entity = temporal_memory.entities.get(ev.entity_id)
                entity_name = entity.canonical_name if entity else ""
                doc = f"[{ev.event_type.upper()}] {entity_name}: {ev.summary}"
                ev_docs.append(doc)
                ev_ids.append(f"{meeting_id}_ev_{eid}")
                ev_metas.append({
                    "meeting_id": meeting_id,
                    "event_type": ev.event_type,
                    "speaker": ev.speaker,
                    "entity_name": entity_name,
                    "start_time": str(ev.start_time),
                })
            if ev_docs:
                self._events.upsert(ids=ev_ids, documents=ev_docs, metadatas=ev_metas)
                counts["events"] = len(ev_docs)

        # ── 3. Meeting directory entry ────────────────────────────────
        title = job.get("title", "Untitled")
        date = job.get("created_at", "")
        participants = job.get("participants", [])
        p_names = ", ".join(
            p.get("name", "?") + f" ({p.get('role', 'Attendee')})" for p in participants
        ) if participants else "Unknown"
        dir_doc = f"Meeting: {title} | Date: {date} | Participants: {p_names}"
        self._meetings.upsert(
            ids=[f"dir_{meeting_id}"],
            documents=[dir_doc],
            metadatas=[{"meeting_id": meeting_id, "date": str(date), "title": title}],
        )
        counts["meeting_dir"] = 1

        logger.info(f"ContextStore indexed meeting {meeting_id}: {counts}")
        return counts

    # ================================================================
    # Retrieval
    # ================================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        scope: str = "all",
        meeting_id: Optional[str] = None,
    ) -> str:
        """
        Semantic search across collections. Returns a formatted text block
        ready for injection into the LLM conversation.

        scope: "all" | "transcript" | "events" | "meetings"
        """
        results_parts: List[str] = []
        where_filter = {"meeting_id": meeting_id} if meeting_id else None

        if scope in ("all", "transcript"):
            hits = self._transcript.query(
                query_texts=[query],
                n_results=min(top_k, 5),
                where=where_filter,
            )
            for doc, meta in zip(hits["documents"][0], hits["metadatas"][0]):
                speaker = meta.get("speaker", "")
                t = meta.get("start_time", "")
                prefix = f"[{t}] [{speaker}] " if speaker else f"[{t}] "
                results_parts.append(f"📝 {prefix}{doc}")

        if scope in ("all", "events"):
            hits = self._events.query(
                query_texts=[query],
                n_results=min(top_k, 5),
                where=where_filter,
            )
            for doc, meta in zip(hits["documents"][0], hits["metadatas"][0]):
                meeting_title = meta.get("meeting_id", "")
                results_parts.append(f"⚡ {doc} (meeting: {meeting_title})")

        if scope in ("all", "meetings"):
            hits = self._meetings.query(
                query_texts=[query],
                n_results=min(top_k, 3),
            )
            for doc in hits["documents"][0]:
                results_parts.append(f"📅 {doc}")

        if not results_parts:
            return "No relevant context found. Try a different query or scope."

        return "\n".join(results_parts)

    def get_meeting_list(self) -> str:
        """Return a compact 1-line-per-meeting directory."""
        all_docs = self._meetings.get()
        if not all_docs["documents"]:
            return "No meetings indexed yet."
        return "\n".join(f"- {doc}" for doc in all_docs["documents"])

    # ================================================================
    # Internal helpers
    # ================================================================

    @staticmethod
    def _chunk_transcript(transcript: str, window: int = 3) -> List[Dict]:
        """
        Split a transcript into overlapping windows of `window` lines.
        Extracts speaker and rough timestamp from bracketed prefixes.
        """
        lines = [l.strip() for l in transcript.split("\n") if l.strip()]
        if not lines:
            return []

        chunks = []
        ts_pattern = re.compile(r"^\[(\d[\d:]+)\]")
        spk_pattern = re.compile(r"^\[[^\]]*\]\s*\[([^\]]+)\]")

        for i in range(0, len(lines), max(1, window - 1)):
            window_lines = lines[i : i + window]
            text = " ".join(window_lines)

            # Extract speaker from first line of window
            spk_match = spk_pattern.match(window_lines[0])
            speaker = spk_match.group(1) if spk_match else ""

            # Extract timestamp from first line
            ts_match = ts_pattern.match(window_lines[0])
            start_time = ts_match.group(1) if ts_match else ""

            chunks.append({
                "text": text,
                "speaker": speaker,
                "start_time": start_time,
            })

        return chunks
