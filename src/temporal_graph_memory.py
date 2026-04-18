"""
Temporal Graph Memory Module (4-Layer Architecture)

Implements an implicit memory system that maps meeting segments to canonical entities
across time without requiring a heavy graph database.

Layers:
Layer 0 - Ingestion
Layer 1 - Universal Event + Entity Extraction (via Qwen 3.5 LLM)
Layer 2 - Entity Canonicalization (Cross-reference via embeddings)
Layer 3 - Temporal Event Graph (Implicit relational edges)
Layer 4 - Temporal Reasoning & Fusion Signals
"""

import json
import os
import uuid
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import numpy as np
from collections import defaultdict
import re as _re

# ─── Filler / Backchannel Detection ──────────────────────────────────────────
FILLER_SET = frozenset({
    "okay", "ok", "yeah", "yep", "yes", "no", "nope", "nah",
    "mm", "mmm", "mm-hmm", "mmhmm", "mhm", "uh-huh", "uh", "huh",
    "hmm", "hm", "ah", "oh", "um", "er", "uh-huh",
    "right", "sure", "cool", "great", "fine", "alright",
    "kay", "'kay", "wow", "so", "well", "anyway",
    "bye", "hi", "hello", "hey", "thanks",
    "exactly", "indeed", "absolutely", "definitely", "fantastic",
    "yup", "agreed", "true", "totally",
})

# Multi-word conversational phrases that carry zero topical substance
FILLER_PHRASES = frozenset({
    "i don't know", "i dont know", "i i don't know",
    "what's that", "whats that", "what is that",
    "that's it", "thats it", "that is it",
    "there it is", "ah there it is", "oh there it is",
    "is it", "isn't it", "isnt it",
    "aye that's a good idea", "that's a good idea", "good idea",
    "yeah that's a good idea",
    "i think so", "i guess so", "i suppose so",
    "let me see", "let's see", "let me think",
    "you know", "you know what i mean",
    "i mean", "i see", "i agree",
    "oh right", "oh yeah", "oh really",
    "play-doh time", "play-doh s",
    "no no", "yeah yeah", "okay okay",
    "alright so", "okay so", "yeah so",
    "mm 'kay", "mm kay",
})

# Entities the LLM commonly hallucinates from filler/conversational segments
DISCARD_ENTITIES = frozenset({
    "acknowledgment", "acknowledgement", "agreement", "response",
    "meeting flow", "affirmation", "confirmation", "greeting",
    "backchannel", "filler", "okay", "yeah", "mm-hmm", "right",
    "sure", "hmm", "uh", "um", "yep", "cool", "great", "alright",
    "previous statement", "something", "discussion",
    "uncertainty", "confusion", "question", "curiosity",
    "i don't know", "what's that", "that's it",
    "good idea", "play-doh",
})


def is_meaningful_segment(text: str, min_words: int = 5) -> bool:
    """High-pass filter: rejects filler-only, phrase-level, and ultra-short segments."""
    cleaned = _re.sub(r'[^\w\s\'-]', '', text.lower()).strip()
    if not cleaned:
        return False

    # Phrase-level check: reject known conversational fragments
    if cleaned in FILLER_PHRASES:
        return False
    # Also check with common punctuation variations stripped
    normalized = _re.sub(r'[^a-z\s]', '', cleaned).strip()
    normalized = _re.sub(r'\s+', ' ', normalized)
    if normalized in FILLER_PHRASES:
        return False

    words = cleaned.split()
    if not words:
        return False
    # Short segments: only keep if they contain non-filler words
    if len(words) < min_words:
        return not all(w.strip(".,!?'\"") in FILLER_SET for w in words)
    # Longer segments must have ≥3 non-filler words
    meaningful = [w for w in words if w.strip(".,!?'\"") not in FILLER_SET]
    return len(meaningful) >= 3


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def _normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """Levenshtein similarity normalized to [0, 1]."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(s1, s2) / max_len)

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback imports matching the style
try:
    from .llm_summarizer import LLMSummarizer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM Summarizer not found, Event Extraction will be mocked.")

try:
    from .topic_classifier import TopicClassifier
    BERT_CLASSIFIER_AVAILABLE = True
except ImportError:
    BERT_CLASSIFIER_AVAILABLE = False
    logger.warning("TopicClassifier not found, will fall back to LLM classification.")


class EventType(Enum):
    PROBLEM = "problem"
    UPDATE = "update"
    DECISION = "decision"
    RISK = "risk"
    IDEA = "idea"
    DEADLINE = "deadline"
    METRIC = "metric"
    DISCUSSION = "discussion"

class NodeType(Enum):
    MEETING = "meeting"
    TOPIC = "topic"
    DECISION = "decision"
    ACTION_ITEM = "action_item"
    PERSON = "person"
    SEGMENT = "segment"


@dataclass
class EntityMemory:
    """
    (Graph Node) Represents a canonical concept tracked across time.
    """
    _id: str
    canonical_name: str
    type: str  # topic | feature | metric | person
    embedding: List[float]
    aliases: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    mention_count: int = 0
    
    # Computed Temporal Signals
    recurrence_score: float = 0.0
    unresolved_score: float = 0.0
    sentiment_trend: float = 0.0
    
    speaker_stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            '_id': self._id,
            'canonical_name': self.canonical_name,
            'type': self.type,
            'embedding': self.embedding,
            'aliases': self.aliases,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'mention_count': self.mention_count,
            'recurrence_score': self.recurrence_score,
            'unresolved_score': self.unresolved_score,
            'sentiment_trend': self.sentiment_trend,
            'speaker_stats': self.speaker_stats
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EntityMemory':
        return cls(
            _id=data['_id'],
            canonical_name=data['canonical_name'],
            type=data.get('type', 'topic'),
            embedding=data.get('embedding', []),
            aliases=data.get('aliases', []),
            first_seen=datetime.fromisoformat(data['first_seen']),
            last_seen=datetime.fromisoformat(data['last_seen']),
            mention_count=data.get('mention_count', 1),
            recurrence_score=data.get('recurrence_score', 0.0),
            unresolved_score=data.get('unresolved_score', 0.0),
            sentiment_trend=data.get('sentiment_trend', 0.0),
            speaker_stats=data.get('speaker_stats', {})
        )


@dataclass
class EventMemory:
    """
    (Temporal Edge) Represents a specific moment in time connected to an entity.
    """
    _id: str
    entity_id: str
    meeting_id: str
    timestamp: datetime
    speaker: str
    event_type: str
    summary: str
    sentiment: float = 0.0
    confidence: float = 0.0
    is_screen_sharing: bool = False
    ocr_text: str = ""
    start_time: float = 0.0  # seconds offset from start of meeting

    def to_dict(self) -> Dict:
        d = {
            '_id': self._id,
            'entity_id': self.entity_id,
            'meeting_id': self.meeting_id,
            'timestamp': self.timestamp.isoformat(),
            'speaker': self.speaker,
            'event_type': self.event_type,
            'summary': self.summary,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'is_screen_sharing': bool(self.is_screen_sharing),
            'start_time': self.start_time,
        }
        if self.ocr_text:
            d['ocr_text'] = self.ocr_text
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'EventMemory':
        return cls(
            _id=data['_id'],
            entity_id=data['entity_id'],
            meeting_id=data['meeting_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            speaker=data.get('speaker', 'Unknown'),
            event_type=data.get('event_type', 'discussion'),
            summary=data.get('summary', ''),
            sentiment=data.get('sentiment', 0.0),
            confidence=data.get('confidence', 0.0),
            is_screen_sharing=data.get('is_screen_sharing', False),
            ocr_text=data.get('ocr_text', ''),
            start_time=data.get('start_time', 0.0)
        )


class TemporalGraphMemory:
    """
    4-Layer Architecture for Cross-Meeting Memory Storage and Retrieval.
    No heavy graph DB required; uses implicit dict-based associations mapped to JSON/SQLite.
    """

    def __init__(
        self,
        storage_path: str = "meeting_memory",
        similarity_threshold: float = 0.75,
        text_analyzer=None
    ):
        self.storage_path = storage_path
        self.similarity_threshold = similarity_threshold
        self.text_analyzer = text_analyzer
        
        if LLM_AVAILABLE:
            self.llm_extractor = LLMSummarizer()
        else:
            self.llm_extractor = None

        # Load fine-tuned BERT topic classifier (preferred over LLM)
        if BERT_CLASSIFIER_AVAILABLE:
            self.topic_classifier = TopicClassifier()
            if self.topic_classifier.is_ready:
                logger.info("Using BERT topic classifier for segment classification")
            else:
                logger.info("BERT model not trained yet, will fall back to LLM")
                self.topic_classifier = None
        else:
            self.topic_classifier = None

        # Data Stores
        self.entities: Dict[str, EntityMemory] = {}
        self.events: Dict[str, EventMemory] = {}
        self.meetings: Dict[str, Dict] = {}

        # Quick Lookup Indexes
        self.events_by_entity: Dict[str, List[str]] = defaultdict(list)
        self.events_by_meeting: Dict[str, List[str]] = defaultdict(list)
        self.entities_by_type: Dict[str, List[str]] = defaultdict(list)
        
        self.db_file = os.path.join(self.storage_path, "graph_db.json")
        os.makedirs(self.storage_path, exist_ok=True)
        self._load_graph()

    def _generate_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _get_embedding(self, text: str) -> List[float]:
        if self.text_analyzer:
            emb = self.text_analyzer.get_embedding(text)
            if emb is not None:
                return emb.tolist() if isinstance(emb, np.ndarray) else emb
        return []

    # ==================== IO persistence ====================
    def _save_graph(self):
        data = {
            'entities': {k: v.to_dict() for k, v in self.entities.items()},
            'events': {k: v.to_dict() for k, v in self.events.items()},
            'meetings': self.meetings
        }
        with open(self.db_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_graph(self):
        if not os.path.exists(self.db_file):
            return
        try:
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                
            for k, v in data.get('entities', {}).items():
                ent = EntityMemory.from_dict(v)
                self.entities[k] = ent
                self.entities_by_type[ent.type].append(k)

            for k, v in data.get('events', {}).items():
                ev = EventMemory.from_dict(v)
                self.events[k] = ev
                self.events_by_entity[ev.entity_id].append(k)
                self.events_by_meeting[ev.meeting_id].append(k)

            self.meetings = data.get('meetings', {})
            logger.info(f"Loaded Graph DB: {len(self.entities)} Entities, {len(self.events)} Events")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")

    # ==================== Meeting Management ====================    
    def create_meeting(self, title: str, date: Optional[datetime] = None, participants: Optional[List[str]] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict] = None) -> str:
        meeting_id = self._generate_id()
        self.meetings[meeting_id] = {
            '_id': meeting_id,
            'date': (date or datetime.now()).isoformat(),
            'title': title,
            'participants': participants or [],
            'tags': tags or [],
            'metadata': metadata or {}
        }
        self._save_graph()
        return meeting_id

    # ==================== Layer 0 & 1 & 2 & 3 Process Pipeline ====================
    def add_segment(
        self, meeting_id: str, text: str, start_time: float, end_time: float,
        speaker: Optional[str] = None, embedding: Optional[np.ndarray] = None,
        importance_score: float = 0.0,
        prev_text: Optional[str] = None, next_text: Optional[str] = None,
    ) -> str:
        """
        Layer 0: Ingestion with optional sliding-window context.
        prev_text / next_text provide surrounding segments for pronoun resolution.
        """
        if not self.llm_extractor:
            return self._generate_id()

        # ── Layer 0 Gatekeeper: reject filler / backchannel segments ──
        if not is_meaningful_segment(text):
            return self._generate_id()

        # ── Sliding Window: build context-enriched text for the LLM ──
        context_parts = []
        if prev_text and is_meaningful_segment(prev_text):
            context_parts.append(f"[Previous] {prev_text.strip()[:150]}")
        context_parts.append(f"[Current] {text.strip()}")
        if next_text and is_meaningful_segment(next_text):
            context_parts.append(f"[Next] {next_text.strip()[:150]}")
        extraction_text = "\n".join(context_parts) if len(context_parts) > 1 else text

        # Step 1: Universal Extraction (LLM)
        timestamp_str = f"{start_time:.1f}s"
        extraction = self.llm_extractor.extract_json_events(extraction_text, speaker, timestamp_str)

        events_data = extraction.get("events", [])
        for ev_data in events_data:
            if ev_data.get("confidence", 0) < 0.4:  # filter hallucinations
                continue
                
            entities = ev_data.get("entities", [])
            for ent in entities:
                ent_val = ent.get("value")
                ent_type = ent.get("type", "topic")
                
                if not ent_val:
                    continue
                
                # Skip discard-listed entities
                if ent_val.lower().strip() in DISCARD_ENTITIES:
                    continue

                # Step 2: Canonicalization (Double-pass: Vector + Levenshtein)
                entity_node = self.resolve_entity(ent_val, ent_type)
                
                # Expand specific speaker tracking
                speaker_val = ev_data.get("speaker", speaker or "Unknown")
                entity_node.speaker_stats[speaker_val] = entity_node.speaker_stats.get(speaker_val, 0) + 1
                entity_node.mention_count += 1
                entity_node.last_seen = datetime.now()

                # Step 3: Event Insertion (Temporal Linking)
                new_event = EventMemory(
                    _id=self._generate_id(),
                    entity_id=entity_node._id,
                    meeting_id=meeting_id,
                    timestamp=datetime.now(),
                    speaker=speaker_val,
                    event_type=ev_data.get("event_type", "discussion"),
                    summary=ev_data.get("summary", ""),
                    sentiment=ev_data.get("sentiment", 0.0),
                    confidence=ev_data.get("confidence", 0.8)
                )

                self.events[new_event._id] = new_event
                self.events_by_entity[new_event.entity_id].append(new_event._id)
                self.events_by_meeting[new_event.meeting_id].append(new_event._id)

                # Step 5: Temporal Signal Computation
                self._compute_temporal_signals(entity_node._id)
                
        self._save_graph()
        return self._generate_id()

    # ==================== Layer 2 Algorithm: Double-Pass Resolution ====================
    def resolve_entity(self, entity_text: str, entity_type: str) -> EntityMemory:
        """
        Double-pass entity resolution:
          Pass 1: Semantic similarity (cosine on embeddings)
          Pass 2: Lexical verification (normalized Levenshtein distance)
          Combined: 0.6 * semantic + 0.4 * lexical
        """
        emb = self._get_embedding(entity_text)
        text_lower = entity_text.lower().strip()

        if not emb:
            # Fallback: pure lexical matching when no embeddings
            for ent_id in self.entities_by_type.get(entity_type, []):
                existing = self.entities[ent_id]
                lex_sim = _normalized_levenshtein_similarity(
                    text_lower, existing.canonical_name.lower().strip()
                )
                # Also check aliases
                best_lex = lex_sim
                for alias in existing.aliases:
                    best_lex = max(best_lex, _normalized_levenshtein_similarity(text_lower, alias.lower()))
                if best_lex > 0.85:  # high lexical threshold for no-embedding fallback
                    if text_lower not in [a.lower() for a in existing.aliases] and text_lower != existing.canonical_name.lower():
                        existing.aliases.append(entity_text)
                    return existing
            return self._create_new_entity(entity_text, entity_type, [])

        # Pass 1: Semantic candidates
        candidates: List[Tuple[EntityMemory, float, float]] = []  # (entity, semantic_sim, lexical_sim)
        for ent_id in self.entities_by_type.get(entity_type, []):
            existing_ent = self.entities[ent_id]
            semantic_sim = 0.0
            if existing_ent.embedding:
                semantic_sim = self._cosine_similarity(emb, existing_ent.embedding)

            # Pass 2: Lexical verification
            lex_sim = _normalized_levenshtein_similarity(
                text_lower, existing_ent.canonical_name.lower().strip()
            )
            # Also check aliases for better lexical matching
            for alias in existing_ent.aliases:
                lex_sim = max(lex_sim, _normalized_levenshtein_similarity(text_lower, alias.lower()))

            # Combined score: semantic-weighted hybrid
            combined = 0.6 * semantic_sim + 0.4 * lex_sim
            candidates.append((existing_ent, combined, lex_sim))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidate, top_combined, top_lex = candidates[0]
            if top_combined > self.similarity_threshold:
                # Merge into existing
                if text_lower not in [a.lower() for a in top_candidate.aliases] and text_lower != top_candidate.canonical_name.lower():
                    top_candidate.aliases.append(entity_text)
                return top_candidate

        return self._create_new_entity(entity_text, entity_type, emb)

    def _create_new_entity(self, entity_text: str, entity_type: str, emb: List[float]) -> EntityMemory:
        new_ent = EntityMemory(
            _id=self._generate_id(),
            canonical_name=entity_text,
            type=entity_type,
            embedding=emb,
            first_seen=datetime.now(),
            last_seen=datetime.now()
        )
        self.entities[new_ent._id] = new_ent
        self.entities_by_type[entity_type].append(new_ent._id)
        return new_ent

    # ==================== Layer 4 / 5: Signals & Inference ====================
    # Decay constant for unresolved score (per day)
    DECAY_LAMBDA = 0.05  # ~14-day half-life

    # Decision state machine transitions
    _DECISION_TRANSITIONS = {
        "decision": "approved",
        "problem":  "contested",
        "risk":     "contested",
        "idea":     "proposed",
        "update":   "approved",
    }

    def _compute_temporal_signals(self, entity_id: str):
        """
        Dynamically updates Recurrence, Unresolved State (with temporal decay),
        Decision State, and Sentiment trends.
        """
        entity = self.entities.get(entity_id)
        if not entity: return

        # A. Recurrence Score (Logarithmic scaling based on frequency)
        entity.recurrence_score = math.log10(1 + entity.mention_count)

        # B & C. Unresolved Score with Temporal Decay & Sentiment Trend
        event_ids = self.events_by_entity.get(entity_id, [])
        timeline_events = sorted([self.events[eid] for eid in event_ids], key=lambda x: x.timestamp)
        
        recent_events = timeline_events[-3:]
        
        if recent_events:
            most_recent = recent_events[-1]
            
            # Base unresolved score from event type
            if most_recent.event_type in ["problem", "risk"]:
                if not any(e.event_type in ["decision", "update"] for e in recent_events):
                    base_score = 1.0
                else:
                    base_score = 0.4
            elif most_recent.event_type == "decision":
                base_score = 0.0
            else:
                base_score = 0.2

            # Apply temporal decay: S(t) = S_base * e^(-λ * Δt_days)
            delta_t = (datetime.now() - most_recent.timestamp).total_seconds() / 86400.0  # days
            entity.unresolved_score = base_score * math.exp(-self.DECAY_LAMBDA * delta_t)

            # Sentiment trend (slope)
            sentiments = [e.sentiment for e in recent_events if e.sentiment != 0.0]
            if len(sentiments) > 1:
                slope = (sentiments[-1] - sentiments[0]) / max(1, len(sentiments) - 1)
                entity.sentiment_trend = slope
            else:
                entity.sentiment_trend = 0.0

    def query_temporal_context(self, query_str: str, top_k: int = 5) -> List[Dict]:
        """
        Layer 4 Reasoning Retrieval with:
          - Alias expansion: matches query against all entities + their aliases
          - Temporal pruning: ranks by recency + importance + sentiment change
          - Provenance: returns source anchors (meeting_id, speaker, timestamp)
        Returns top_k most relevant context items across all matching entities.
        """
        emb = self._get_embedding(query_str)
        query_lower = query_str.lower().strip()

        # --- Step 1: Alias-Expanded Entity Matching ---
        matched_entities = []
        for ent in self.entities.values():
            # Semantic match
            semantic_sim = 0.0
            if emb and ent.embedding:
                semantic_sim = self._cosine_similarity(emb, ent.embedding)

            # Lexical match against canonical name + all aliases
            lex_sim = _normalized_levenshtein_similarity(query_lower, ent.canonical_name.lower())
            for alias in ent.aliases:
                lex_sim = max(lex_sim, _normalized_levenshtein_similarity(query_lower, alias.lower()))

            combined = 0.6 * semantic_sim + 0.4 * lex_sim
            if combined > 0.45:  # lower threshold for broader retrieval
                matched_entities.append((ent, combined))

        if not matched_entities:
            return []

        # --- Step 2: Collect all events from matched entities ---
        all_scored_events = []
        now = datetime.now()
        for ent, entity_sim in matched_entities:
            event_ids = self.events_by_entity.get(ent._id, [])
            for eid in event_ids:
                ev = self.events.get(eid)
                if not ev:
                    continue

                # Temporal pruning score: recency + importance + sentiment
                days_ago = (now - ev.timestamp).total_seconds() / 86400.0
                recency_score = math.exp(-0.05 * days_ago)  # same decay as unresolved

                # Importance: decisions and problems rank higher
                importance_map = {
                    'decision': 1.0, 'problem': 0.9, 'risk': 0.85,
                    'idea': 0.7, 'deadline': 0.8, 'metric': 0.6,
                    'update': 0.5, 'discussion': 0.3,
                }
                importance = importance_map.get(ev.event_type, 0.3)

                # Sentiment change: high absolute sentiment = more noteworthy
                sentiment_signal = abs(ev.sentiment) * 0.3

                # Combined pruning score
                pruning_score = (
                    0.35 * recency_score +
                    0.35 * importance +
                    0.15 * sentiment_signal +
                    0.15 * entity_sim  # entity relevance
                )

                all_scored_events.append((ev, ent, pruning_score))

        # --- Step 3: Rank and select top_k ---
        all_scored_events.sort(key=lambda x: x[2], reverse=True)
        top_events = all_scored_events[:top_k]

        # --- Step 4: Format with provenance ---
        results = []
        for ev, ent, score in top_events:
            meeting_title = self.meetings.get(ev.meeting_id, {}).get('title', ev.meeting_id[:8])
            # Format timestamp as MM:SS
            minutes = int(ev.start_time // 60)
            seconds = int(ev.start_time % 60)
            time_str = f"{minutes}:{seconds:02d}"

            results.append({
                'entity': ent.canonical_name,
                'entity_type': ent.type,
                'event_type': ev.event_type,
                'summary': ev.summary,
                'speaker': ev.speaker,
                'meeting_title': meeting_title,
                'meeting_id': ev.meeting_id,
                'timestamp': time_str,
                'date': ev.timestamp.isoformat(),
                'sentiment': ev.sentiment,
                'unresolved_score': ent.unresolved_score,
                'relevance_score': round(score, 3),
                # Markdown citation for LLM injection
                'citation': f"({meeting_title}, {ev.speaker}, {time_str})",
            })

        return results

    def format_context_for_llm(self, query: str, top_k: int = 5) -> str:
        """
        Formats cross-meeting context as a concise, cited block for LLM injection.
        Designed to be prepended to summarizer/chatbot prompts.
        """
        context_items = self.query_temporal_context(query, top_k=top_k)
        if not context_items:
            return ""

        lines = ["[Cross-Meeting Context]"]
        for item in context_items:
            state = ""
            if item['unresolved_score'] > 0.6:
                state = " ⚠️ UNRESOLVED"
            elif item['event_type'] == 'decision':
                state = " ✅ DECIDED"

            lines.append(
                f"- [{item['event_type'].upper()}] {item['entity']}: "
                f"{item['summary'][:120]}{state} "
                f"{item['citation']}"
            )

        lines.append("")
        return "\n".join(lines)

    # ==================== Backward Compatability Adatper ====================
    def get_context_for_segment(self, text: str, embedding: Optional[np.ndarray] = None, current_meeting_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Adapter to ensure `fusion_layer.py` does not break.
        Maps the new entity structure back into the old expected dict format, while providing the new temporal signals.
        """
        context = {
            'related_topics': [],
            'related_decisions': [],
            'open_action_items': [],
            'discussion_history': [],
            'context_score': 0.0,
            'cross_meeting_entities': [],
            
            # Exported temporal signals specifically for fusion layer
            'signals': {
                'recurrence_score': 0.0,
                'unresolved_score': 0.0,
                'sentiment_trend': 0.0,
                'authority_score': 0.0
            }
        }
        
        # Use existing vector search over canonical entities
        emb = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        if not emb:
            if self.text_analyzer:
                arr = self.text_analyzer.get_embedding(text)
                emb = arr.tolist() if isinstance(arr, np.ndarray) else arr
        
        if not emb:
            return context
            
        candidates = []
        for ent in self.entities.values():
            if ent.embedding:
                sim = self._cosine_similarity(emb, ent.embedding)
                candidates.append((ent, sim))
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Gather context
        highest_score = 0.0
        signal_agg = {'recurrence_score': 0.0, 'unresolved_score': 0.0, 'sentiment_trend': 0.0}

        for ent, sim in candidates[:3]:
            if sim < 0.70:
                continue
                
            highest_score = max(highest_score, sim)
            signal_agg['recurrence_score'] = max(signal_agg['recurrence_score'], ent.recurrence_score)
            signal_agg['unresolved_score'] = max(signal_agg['unresolved_score'], ent.unresolved_score)
            # average trends
            signal_agg['sentiment_trend'] += ent.sentiment_trend
            
            context['cross_meeting_entities'].append({
                'name': ent.canonical_name,
                'type': ent.type,
                'mentions': ent.mention_count,
                'similarity': sim
            })

            # Check previous events for decisions/actions
            events = sorted([self.events[eid] for eid in self.events_by_entity.get(ent._id, [])], key=lambda x: x.timestamp)
            for ev in events:
                if ev.meeting_id == current_meeting_id:
                    continue # only cross-meeting
                
                if ev.event_type == 'decision':
                    context['related_decisions'].append({
                        'decision': ev.summary,
                        'meeting_id': ev.meeting_id,
                        'date': ev.timestamp.isoformat(),
                        'related_topic': ent.canonical_name
                    })
                elif ev.event_type in ['problem', 'risk'] and ent.unresolved_score > 0.5:
                    # Treat unresolved problems as things needing action
                    context['open_action_items'].append({
                        'action': ev.summary,
                        'assignee': 'Auto-tracked problem',
                        'due_date': 'Pending',
                        'meeting_id': ev.meeting_id
                    })
                elif ent.type == 'topic':
                    context['related_topics'].append({
                        'topic': ent.canonical_name,
                        'meeting_id': ev.meeting_id,
                        'meeting_date': ev.timestamp.isoformat(),
                        'similarity': sim
                    })

        context['context_score'] = highest_score
        
        # Finalize signals average if multiple matching entities
        count = len(context['cross_meeting_entities']) or 1
        signal_agg['sentiment_trend'] /= count
        context['signals'].update(signal_agg)
        
        return context

    # ==================== Backward Compatability Adatper ====================
    def get_statistics(self) -> Dict[str, int]:
        topics = len(self.entities_by_type.get('topic', []))
        decisions = len([e for e in self.events.values() if e.event_type == 'decision'])
        actions = len([e for e in self.events.values() if e.event_type in ['problem', 'risk'] and e.confidence > 0.5])
        return {
            'meetings': len(self.meetings),
            'topics': topics,
            'decisions': decisions,
            'action_items': actions
        }
        
    def save(self):
        self._save_graph()
        
    def get_all_meetings(self) -> List[Any]:
        # Fast mock returning objects with id and content for UI
        class MockMtg:
            def __init__(self, m):
                self.id = m['_id']
                self.content = m['title']
                self.timestamp = datetime.fromisoformat(m['date'])
                self.metadata = {'tags': m.get('tags', [])}
        return [MockMtg(m) for m in self.meetings.values()]
        
    def get_meeting_summary(self, meeting_id: str) -> Dict[str, List]:
        events = self.events_by_meeting.get(meeting_id, [])
        ev_objs = [self.events[eid] for eid in events]
        return {
            'segments': ev_objs, # loose mapping
            'topics': [e for e in ev_objs if e.event_type == 'discussion'],
            'decisions': [e for e in ev_objs if e.event_type == 'decision'],
            'action_items': [e for e in ev_objs if e.event_type in ['problem', 'risk']]
        }
        
    # ── Topic Label Extraction for Cross-Meeting Matching ──────────────────
    _topic_cache: Dict[str, List[str]] = {}  # cache to avoid re-extracting

    def _extract_topic_labels(self, text: str, idx: int = 0, all_segments: list = None) -> List[str]:
        """
        Extract 2-4 word topic keywords from a segment text.
        Uses LLM when available, falls back to content-word extraction.
        Returns a list of short topic labels suitable for entity resolution.
        """
        # Check cache
        cache_key = text.strip()[:100]
        if cache_key in self._topic_cache:
            return self._topic_cache[cache_key]

        labels = []

        # Try LLM extraction (batch-friendly)
        if self.llm_extractor and hasattr(self.llm_extractor, '_call_ollama'):
            try:
                prompt = (
                    "/no_think\n"
                    "Extract the key topic keywords (2-4 words each) from this meeting transcript segment.\n"
                    "Return ONLY a comma-separated list of short topic labels.\n"
                    "Focus on concrete nouns, features, components, technologies, or decisions.\n"
                    "Do NOT include filler words, acknowledgments, or abstract labels.\n"
                    "If the segment has no meaningful topics, return: NONE\n\n"
                    f"Segment: {text.strip()[:200]}\n\n"
                    "Topics:"
                )
                raw_response = self.llm_extractor._call_ollama(prompt, format_json=False, temperature=0.1)
                raw_response = raw_response.strip()

                if raw_response.upper() != "NONE" and raw_response:
                    # Parse comma-separated labels
                    for label in raw_response.split(","):
                        clean = label.strip().strip('"\'.-')
                        if clean and len(clean) > 2 and len(clean) < 60:
                            labels.append(clean)
            except Exception as e:
                logger.debug(f"LLM topic extraction failed: {e}")

        # Fallback: content-word extraction (no LLM needed)
        if not labels:
            labels = self._extract_content_keywords(text)

        self._topic_cache[cache_key] = labels
        return labels

    def _extract_content_keywords(self, text: str) -> List[str]:
        """
        Fallback keyword extractor: extracts meaningful noun phrases
        by filtering out stopwords and filler words.
        """
        # Extended stopwords for meeting context
        stopwords = FILLER_SET | frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can",
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our", "their",
            "this", "that", "these", "those", "which", "what", "who",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "about", "into", "through", "during", "before", "after",
            "and", "but", "or", "not", "if", "then", "than", "too",
            "very", "just", "also", "like", "think", "know", "want",
            "going", "gonna", "got", "get", "go", "come", "make",
            "thing", "things", "stuff", "lot", "kind", "sort",
            "really", "actually", "basically", "probably", "maybe",
            "something", "anything", "everything", "nothing",
        })

        words = _re.sub(r'[^\w\s-]', '', text.lower()).split()
        content_words = [w for w in words if w not in stopwords and len(w) > 2]

        if not content_words:
            return []

        # Build short phrases from consecutive content words (max 4 words)
        # Simple approach: take the first meaningful chunk
        result = " ".join(content_words[:4])
        return [result] if result else []

    def ingest_meeting_results(self, meeting_id: str, scored_segments: List, importance_threshold: float = 0.4, clear: bool = False):
        """
        Batch ingest of scored SegmentFeatures into the temporal graph.
        Creates EntityMemory (topic nodes) and EventMemory (edges) from scored segments.
        Falls back to direct embedding-based canonicalization when LLM is not used.
        """
        # Debug: log score distribution
        scores = []
        for s in scored_segments:
            if isinstance(s, dict):
                scores.append(s.get('fused_score', s.get('score', 0.0)))
            else:
                scores.append(getattr(s, 'fused_score', 0.0))
        if scores:
            logger.info(
                f"ingest_meeting_results: {len(scores)} segments, "
                f"score range: {min(scores):.4f}-{max(scores):.4f}, "
                f"above {importance_threshold}: {sum(1 for s in scores if s >= importance_threshold)}, "
                f"threshold: {importance_threshold}"
            )
        if clear:
            # Drop older events for this meeting
            to_remove = list(self.events_by_meeting.get(meeting_id, []))
            for eid in to_remove:
                if eid in self.events:
                    ent_id = self.events[eid].entity_id
                    if eid in self.events_by_entity.get(ent_id, []):
                        self.events_by_entity[ent_id].remove(eid)
                    del self.events[eid]
            self.events_by_meeting[meeting_id] = []
            
        # Process each scored segment and insert into the graph
        ingested_events = []
        ingested_topics = []
        ingested_decisions = []
        ingested_actions = []
        # ── LLM batch classification ──────────────────────────────────────────
        # Collect candidate texts for classification (only those above threshold)
        candidate_texts: List[Tuple[int, str, str]] = []  # (index, text, speaker)
        for idx, seg in enumerate(scored_segments):
            if isinstance(seg, dict):
                fs = seg.get('fused_score', seg.get('score', 0.0))
                txt = seg.get('text', '')
                spk = seg.get('speaker', None) or 'Unknown'
            else:
                fs = getattr(seg, 'fused_score', 0.0)
                txt = getattr(seg, 'text', '')
                spk = getattr(seg, 'speaker', None) or 'Unknown'
            if fs >= importance_threshold and txt and txt.strip():
                candidate_texts.append((idx, txt.strip()[:200], spk))

        classifications: Dict[int, str] = {}
        valid_types = {'decision', 'problem', 'update', 'idea', 'deadline', 'metric', 'discussion', 'risk'}

        # ── BERT-based classification (replaces LLM) ──────────────────────────
        if self.topic_classifier and self.topic_classifier.is_ready and candidate_texts:
            texts_only = [item[1] for item in candidate_texts]
            results = self.topic_classifier.classify_batch(texts_only)
            for i, (label, confidence) in enumerate(results):
                idx = candidate_texts[i][0]
                classifications[idx] = label if label in valid_types else 'discussion'
            logger.info(f"BERT classified {len(classifications)}/{len(candidate_texts)} segments")
        elif self.llm_extractor and candidate_texts:
            # Fallback to LLM if BERT model is not available
            BATCH_SIZE = 10
            if not self.llm_extractor.is_ready:
                self.llm_extractor._check_connection()
            if self.llm_extractor.is_ready:
                import re
                for batch_start in range(0, len(candidate_texts), BATCH_SIZE):
                    batch = candidate_texts[batch_start:batch_start + BATCH_SIZE]
                    numbered = "\n".join(
                        f"{i+1}. [{item[2]}]: {item[1]}" for i, item in enumerate(batch)
                    )
                    prompt = (
                        "/no_think\n"
                        "Classify each meeting transcript line into exactly ONE type: "
                        "decision, problem, discussion, update, idea, risk\n\n"
                        "Rules:\n"
                        "- decision: a choice or agreement was made\n"
                        "- problem: an issue, bug, blocker, or concern raised\n"
                        "- update: status report or progress info\n"
                        "- idea: a suggestion or proposal\n"
                        "- risk: a potential future issue\n"
                        "- discussion: general conversation (default)\n\n"
                        f"Lines:\n{numbered}\n\n"
                        "Reply with ONLY a numbered list like:\n"
                        "1. decision\n2. problem\n3. discussion\n\n"
                        f"Classify all {len(batch)} lines:"
                    )
                    try:
                        raw = self.llm_extractor._call_ollama(prompt, format_json=False, temperature=0.1)
                        lines = raw.strip().split('\n')
                        for line in lines:
                            m = re.match(r'\d+\.\s*(\w+)', line.strip())
                            if m:
                                line_num = int(re.match(r'(\d+)', line.strip()).group(1)) - 1
                                if 0 <= line_num < len(batch):
                                    t = m.group(1).lower().strip()
                                    classifications[batch[line_num][0]] = t if t in valid_types else 'discussion'
                    except Exception as e:
                        logger.warning(f"LLM batch classification failed: {e}")
                logger.info(f"LLM classified {len(classifications)}/{len(candidate_texts)} segments")
        # ── End classification ─────────────────────────────────────────────────

        for idx, seg in enumerate(scored_segments):
            if isinstance(seg, dict):
                fused_score = seg.get('fused_score', seg.get('score', 0.0))
                text = seg.get('text', '')
                speaker = seg.get('speaker', None) or 'Unknown'
                start_time = seg.get('start_time', seg.get('start', 0.0))
                unresolved = seg.get('unresolved_score', 0.0)
                semantic = seg.get('semantic_score', 0.0)
                emb_raw = seg.get('text_embedding', None)
            else:
                fused_score = getattr(seg, 'fused_score', 0.0)
                text = getattr(seg, 'text', '')
                speaker = getattr(seg, 'speaker', None) or 'Unknown'
                start_time = getattr(seg, 'start_time', 0.0)
                unresolved = getattr(seg, 'unresolved_score', 0.0)
                semantic = getattr(seg, 'semantic_score', 0.0)
                emb_raw = getattr(seg, 'text_embedding', None)

            if fused_score < importance_threshold:
                continue

            if not text or not text.strip():
                continue

            # Gatekeeper: skip filler / backchannel segments
            if not is_meaningful_segment(text):
                continue

            # Get embedding (prefer pre-computed from SegmentFeatures)
            emb = emb_raw
            if emb is None and self.text_analyzer:
                arr = self.text_analyzer.get_embedding(text)
                emb = arr.tolist() if isinstance(arr, np.ndarray) else (arr or [])
            elif isinstance(emb, np.ndarray):
                emb = emb.tolist()
            elif emb is None:
                emb = []

            # Event type will be filled by LLM classification (computed below)
            event_type = classifications.get(idx, 'discussion')

            # Map event type → entity node type for graph coloring
            entity_type_map = {
                'decision': 'decision', 'problem': 'problem', 'risk': 'problem',
                'idea': 'topic', 'update': 'topic', 'deadline': 'topic',
                'metric': 'topic', 'discussion': 'topic',
            }
            entity_type = entity_type_map.get(event_type, 'topic')

            # ── Extract short topic labels for cross-meeting matching ──
            # Instead of using raw text: extract 2-4 word keywords via LLM
            raw = text.strip()
            topic_labels = self._extract_topic_labels(raw, idx, scored_segments)

            # Summary text for events
            if len(raw) > 200:
                scut = raw[:200].rfind(' ')
                summary_text = raw[:scut] + '…' if scut > 40 else raw[:200] + '…'
            else:
                summary_text = raw

            # Create entity + event for each extracted topic label
            created_any = False
            for topic_label in topic_labels:
                # Skip discard-listed entities
                if topic_label.lower().strip() in DISCARD_ENTITIES:
                    continue
                if topic_label.lower().strip() in FILLER_SET:
                    continue
                tl_stripped = _re.sub(r'[^\w\s]', '', topic_label.lower()).strip()
                if len(tl_stripped) < 3:
                    continue
                tl_words = tl_stripped.split()
                if tl_words and all(w in FILLER_SET for w in tl_words):
                    continue

                entity_node = self.resolve_entity(topic_label, entity_type)
                entity_node.speaker_stats[speaker] = entity_node.speaker_stats.get(speaker, 0) + 1
                entity_node.mention_count += 1
                entity_node.last_seen = datetime.now()

                new_event = EventMemory(
                    _id=self._generate_id(),
                    entity_id=entity_node._id,
                    meeting_id=meeting_id,
                    timestamp=datetime.now(),
                    speaker=speaker,
                    event_type=event_type,
                    summary=summary_text,
                    sentiment=getattr(seg, 'tonal_score', 0.0) - 0.5,
                    confidence=fused_score,
                    is_screen_sharing=getattr(seg, 'is_screen_sharing', False),
                    ocr_text=getattr(seg, 'ocr_text', ''),
                    start_time=float(start_time)
                )
                self.events[new_event._id] = new_event
                self.events_by_entity[entity_node._id].append(new_event._id)
                self.events_by_meeting[meeting_id].append(new_event._id)
                self._compute_temporal_signals(entity_node._id)

                ingested_events.append(new_event)
                if event_type == 'decision':
                    ingested_decisions.append(new_event)
                elif event_type == 'problem':
                    ingested_actions.append(new_event)
                else:
                    ingested_topics.append(new_event)
                created_any = True

        if ingested_events:
            self._save_graph()
            logger.info(
                f"ingest_meeting_results: meeting={meeting_id} "
                f"events={len(ingested_events)} topics={len(ingested_topics)} "
                f"decisions={len(ingested_decisions)} actions={len(ingested_actions)}"
            )

        return {
            'segments': ingested_events,
            'topics': ingested_topics,
            'decisions': ingested_decisions,
            'action_items': ingested_actions
        }

    def get_open_action_items(self) -> List[Dict]:
        items = []
        for ent in self.entities.values():
            if ent.unresolved_score > 0.5:
                events = self.events_by_entity.get(ent._id, [])
                if events:
                    last_ev = self.events[events[-1]]
                    meeting_info = self.meetings.get(last_ev.meeting_id, {})
                    items.append({
                        'action_id': last_ev._id,
                        'action': last_ev.summary or ent.canonical_name,
                        'priority': 'high' if ent.unresolved_score > 0.8 else 'medium',
                        'assignee': 'Auto-tracked',
                        'due_date': 'Pending',
                        'date': last_ev.timestamp.isoformat(),
                        'meeting_id': last_ev.meeting_id,
                        'meeting_title': meeting_info.get('title', 'Unknown'),
                        'related_topic': ent.canonical_name
                    })
        return items

    def update_action_status(self, action_id: str, status: str):
        """Mark an event/action as resolved by removing its entity's unresolved flag."""
        event = self.events.get(action_id)
        if event:
            entity = self.entities.get(event.entity_id)
            if entity and status == 'done':
                entity.unresolved_score = 0.0
                self._save_graph()
                logger.info(f"Marked action {action_id} as done (entity: {entity.canonical_name})")
        
    def find_cross_meeting_threads(self, min_meetings: int = 2, similarity_threshold: float = 0.5) -> List[Dict]:
        total_meetings = max(len(self.meetings), 1)
        threads = []
        for ent in self.entities.values():
            # ── Query-time noise filter ──────────────────────────────────
            name_lower = ent.canonical_name.lower().strip()
            # 1. Static blacklist: common LLM hallucinations from fillers
            if name_lower in DISCARD_ENTITIES:
                continue
            # 2. Too short to be meaningful (single word fillers that slipped through)
            stripped = _re.sub(r'[^\w\s]', '', name_lower).strip()
            if len(stripped) < 3 or stripped in FILLER_SET:
                continue
            # 3. Check if all words in the entity name are fillers
            words = stripped.split()
            if words and all(w in FILLER_SET for w in words):
                continue

            mtgs = set(self.events[eid].meeting_id for eid in self.events_by_entity.get(ent._id, []))
            if len(mtgs) < min_meetings:
                continue

            # 4. Ubiquity filter: if entity appears in >90% of meetings
            #    but has very short name (<15 chars), it's likely noise
            if len(mtgs) / total_meetings > 0.9 and len(name_lower) < 15:
                continue

            events = sorted(
                [self.events[eid] for eid in self.events_by_entity.get(ent._id, [])],
                key=lambda x: x.timestamp
            )
            threads.append({
                'thread_id': ent._id,
                'label': ent.canonical_name,
                'meeting_count': len(mtgs),
                'first_seen': events[0].timestamp.isoformat(),
                'last_seen': events[-1].timestamp.isoformat(),
                'keywords': ent.aliases[:5],
                'appearances': [
                    {
                        'date': ev.timestamp.isoformat(),
                        'meeting_title': self.meetings.get(ev.meeting_id, {}).get('title', 'Unknown'),
                        'topic': ev.summary,
                        'keywords': []
                    } for ev in events
                ]
            })
        # Sort by meeting count descending so most connected threads appear first
        threads.sort(key=lambda t: t['meeting_count'], reverse=True)
        return threads

    def get_context_for_text(self, text: str, top_k: int = 5) -> List[Dict]:
        """Returns context items with provenance for a given text query."""
        return self.query_temporal_context(text, top_k=top_k)

    # Mock legacy tracking attributes expected by UI/Threads
    @property
    def nodes(self): return {}
    @property
    def nodes_by_type(self): return {}
    def get_edges_from(self, *args, **kwargs): return []
    def add_topic(self, *args, **kwargs): return self._generate_id()
    def add_decision(self, *args, **kwargs): return self._generate_id()
    def add_action_item(self, *args, **kwargs): return self._generate_id()
    def get_node(self, *args, **kwargs): return None
