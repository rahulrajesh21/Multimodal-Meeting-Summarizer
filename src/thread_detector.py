"""
Thread Detector — Cross-Meeting Topic Clustering.

Wraps TemporalGraphMemory to provide:
- detect_threads(): cluster semantically similar topics from different meetings
- get_thread_for_segment(): check if a live segment belongs to a known thread
- generate_thread_annotation(): human-readable "⚠️ Revisited" label for summaries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThreadAppearance:
    """One occurrence of a thread (one meeting)."""
    meeting_id: str
    meeting_title: str
    date: str            # ISO-format string (keeps JSON-safe)
    topic: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class Thread:
    """A recurring topic cluster that spans multiple meetings."""
    thread_id: str
    canonical_label: str
    first_seen: str      # ISO-format string
    last_seen: str
    meeting_count: int
    keywords: List[str]
    appearances: List[ThreadAppearance]  # chronological

    def days_since_first(self) -> int:
        try:
            delta = datetime.now() - datetime.fromisoformat(self.first_seen)
            return delta.days
        except Exception:
            return 0

    def days_since_last(self) -> int:
        try:
            delta = datetime.now() - datetime.fromisoformat(self.last_seen)
            return delta.days
        except Exception:
            return 0


class ThreadDetector:
    """
    Detects recurring discussion threads across meetings stored in TemporalGraphMemory.

    Usage::

        detector = ThreadDetector(temporal_memory)
        threads  = detector.detect_threads()

        # For a live segment during analysis:
        thread = detector.get_thread_for_segment(seg_text, embedding)
        if thread:
            label = detector.generate_thread_annotation(thread)
    """

    def __init__(
        self,
        temporal_memory,
        similarity_threshold: float = 0.70,
        min_occurrences: int = 2,
    ):
        """
        Args:
            temporal_memory: TemporalGraphMemory instance.
            similarity_threshold: Cosine similarity required to merge two topic nodes.
            min_occurrences: Minimum unique meetings a thread must span to be reported.
        """
        self.memory = temporal_memory
        self.similarity_threshold = similarity_threshold
        self.min_occurrences = min_occurrences
        self._cached_threads: Optional[List[Thread]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_threads(self, force_refresh: bool = False) -> List[Thread]:
        """
        Detect and return all cross-meeting threads.

        Results are cached until force_refresh=True is passed (or until
        ingest_meeting_results is called on the memory).

        Returns:
            List of Thread objects, sorted most-recently-active first.
        """
        if self._cached_threads is not None and not force_refresh:
            return self._cached_threads

        raw = self.memory.find_cross_meeting_threads(
            min_meetings=self.min_occurrences,
            similarity_threshold=self.similarity_threshold,
        )

        threads = []
        for t in raw:
            appearances = [
                ThreadAppearance(
                    meeting_id=a["meeting_id"],
                    meeting_title=a["meeting_title"],
                    date=a["date"],
                    topic=a["topic"],
                    keywords=a.get("keywords", []),
                )
                for a in t.get("appearances", [])
            ]
            threads.append(
                Thread(
                    thread_id=t["thread_id"],
                    canonical_label=t["label"],
                    first_seen=t["first_seen"],
                    last_seen=t["last_seen"],
                    meeting_count=t["meeting_count"],
                    keywords=t.get("keywords", []),
                    appearances=appearances,
                )
            )

        self._cached_threads = threads
        logger.info(f"ThreadDetector: detected {len(threads)} threads")
        return threads

    def invalidate_cache(self):
        """Call after a new meeting is ingested so threads are recomputed."""
        self._cached_threads = None

    def get_thread_for_segment(
        self,
        segment_text: str,
        text_embedding: Optional[np.ndarray] = None,
    ) -> Optional[Thread]:
        """
        Check whether a segment belongs to an existing cross-meeting thread.

        Matching strategy:
        1. If embedding is available, compute cosine similarity against each
           thread's topic embeddings in memory (via TemporalGraphMemory nodes).
        2. Fallback: keyword overlap (Jaccard) between segment words and thread keywords.

        Returns the best-matching Thread, or None if no match above threshold.
        """
        threads = self.detect_threads()
        if not threads:
            return None

        seg_lower = segment_text.lower()

        best_thread: Optional[Thread] = None
        best_score: float = 0.0

        for thread in threads:
            score = 0.0

            if text_embedding is not None:
                # Embedding similarity: compare against entity embeddings in memory
                for appearance in thread.appearances:
                    topic_score = self._embedding_similarity_to_memory_topics(
                        text_embedding, appearance.meeting_id, appearance.topic
                    )
                    score = max(score, topic_score)
            else:
                # Keyword Jaccard fallback
                seg_words = set(w for w in seg_lower.split() if len(w) > 3)
                thread_words = set(thread.keywords)
                if thread_words:
                    intersection = seg_words & thread_words
                    union = seg_words | thread_words
                    score = len(intersection) / len(union) if union else 0.0

            if score >= self.similarity_threshold and score > best_score:
                best_score = score
                best_thread = thread

        return best_thread

    def generate_thread_annotation(
        self,
        thread: Thread,
        current_meeting_date: Optional[datetime] = None,
    ) -> str:
        """
        Produce a short human-readable annotation for a segment that belongs to a thread.

        Examples:
            "⚠️ Revisited topic — first raised 5 days ago in 'Sprint Review'"
            "⚠️ Recurring issue — discussed across 3 meetings (security, backend)"

        Args:
            thread: The Thread this segment belongs to.
            current_meeting_date: Date of the current meeting (default: now).

        Returns:
            Annotation string ready for display in UI or LLM prompt.
        """
        if not thread.appearances:
            return ""

        first = thread.appearances[0]
        try:
            first_dt = datetime.fromisoformat(first.date)
            reference = current_meeting_date or datetime.now()
            days_ago = (reference - first_dt).days
            if days_ago == 0:
                when_str = "earlier today"
            elif days_ago == 1:
                when_str = "yesterday"
            else:
                when_str = f"{days_ago} days ago"
        except Exception:
            when_str = "a previous meeting"

        top_kws = ", ".join(thread.keywords[:4]) if thread.keywords else ""
        kw_suffix = f" ({top_kws})" if top_kws else ""

        if thread.meeting_count >= 3:
            return (
                f"⚠️ Recurring issue — discussed across {thread.meeting_count} meetings"
                f"{kw_suffix}"
            )

        return (
            f"⚠️ Revisited topic — first raised {when_str} "
            f"in '{first.meeting_title}'"
            f"{kw_suffix}"
        )

    def get_threads_for_summary(
        self,
        meeting_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Return threads that include the given meeting_id (for LLM prompt injection).

        Returns a list of dicts suitable for injecting into summarize() as
        temporal_context['threads'].
        """
        threads = self.detect_threads()
        results = []
        for thread in threads:
            meeting_ids = {a.meeting_id for a in thread.appearances}
            if meeting_id not in meeting_ids:
                continue
            # Only include if this thread has OTHER meetings too
            other_appearances = [
                a for a in thread.appearances if a.meeting_id != meeting_id
            ]
            if not other_appearances:
                continue
            results.append({
                "label": thread.canonical_label,
                "first_seen": thread.first_seen,
                "first_meeting": thread.appearances[0].meeting_title,
                "meeting_count": thread.meeting_count,
                "keywords": thread.keywords,
                "annotation": self.generate_thread_annotation(thread),
            })
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embedding_similarity_to_memory_topics(
        self,
        query_embedding: np.ndarray,
        meeting_id: str,
        topic_text: str,
    ) -> float:
        """
        Find the best cosine similarity between query_embedding and entity embeddings
        in memory that are linked to the given meeting_id.
        Uses the new EntityMemory-based API (entities_by_type + events_by_entity).
        """
        best = 0.0
        # Iterate topic-type entities in memory
        for ent_id in self.memory.entities_by_type.get('topic', []):
            entity = self.memory.entities.get(ent_id)
            if not entity or not entity.embedding:
                continue
            # Check whether any event for this entity came from the target meeting
            event_ids = self.memory.events_by_entity.get(ent_id, [])
            in_meeting = any(
                self.memory.events[eid].meeting_id == meeting_id
                for eid in event_ids
                if eid in self.memory.events
            )
            if not in_meeting:
                continue
            entity_emb = np.array(entity.embedding)
            sim = self._cosine(query_embedding, entity_emb)
            if sim > best:
                best = sim
        return best

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
