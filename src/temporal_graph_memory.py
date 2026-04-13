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
    mention_count: int = 1
    
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

    def to_dict(self) -> Dict:
        return {
            '_id': self._id,
            'entity_id': self.entity_id,
            'meeting_id': self.meeting_id,
            'timestamp': self.timestamp.isoformat(),
            'speaker': self.speaker,
            'event_type': self.event_type,
            'summary': self.summary,
            'sentiment': self.sentiment,
            'confidence': self.confidence
        }

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
            confidence=data.get('confidence', 0.0)
        )


class TemporalGraphMemory:
    """
    4-Layer Architecture for Cross-Meeting Memory Storage and Retrieval.
    No heavy graph DB required; uses implicit dict-based associations mapped to JSON/SQLite.
    """

    def __init__(
        self,
        storage_path: str = "meeting_memory",
        similarity_threshold: float = 0.82,
        text_analyzer=None
    ):
        self.storage_path = storage_path
        self.similarity_threshold = similarity_threshold
        self.text_analyzer = text_analyzer
        
        if LLM_AVAILABLE:
            self.llm_extractor = LLMSummarizer()
        else:
            self.llm_extractor = None

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
    def add_segment(self, meeting_id: str, text: str, start_time: float, end_time: float, speaker: Optional[str] = None, embedding: Optional[np.ndarray] = None, importance_score: float = 0.0) -> str:
        """
        Layer 0: Ingestion. Also backward compatible with app.py.
        Triggers extraction & canonicalization pipeline natively.
        Returns a mock segment_id since we discarded segments as explicit graph nodes.
        """
        if not self.llm_extractor:
            return self._generate_id()

        # Step 1: Universal Extraction (LLM)
        timestamp_str = f"{start_time:.1f}s"
        extraction = self.llm_extractor.extract_json_events(text, speaker, timestamp_str)

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

                # Step 2: Canonicalization (Vector search & Merge)
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

    # ==================== Layer 2 Algorithm: Resolution ====================
    def resolve_entity(self, entity_text: str, entity_type: str) -> EntityMemory:
        """
        Resolves whether a spoken concept matches an existing canonical entity.
        If similarity > 0.82, merge it. Overwise, create a new one.
        """
        emb = self._get_embedding(entity_text)
        if not emb:
            return self._create_new_entity(entity_text, entity_type, [])

        candidates: List[Tuple[EntityMemory, float]] = []
        for ent_id in self.entities_by_type.get(entity_type, []):
            existing_ent = self.entities[ent_id]
            if existing_ent.embedding:
                sim = self._cosine_similarity(emb, existing_ent.embedding)
                candidates.append((existing_ent, sim))

        if candidates:
            # sort ascending
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidate, top_sim = candidates[0]
            if top_sim > self.similarity_threshold:
                # Merge into existing!
                if entity_text.lower() not in [a.lower() for a in top_candidate.aliases] and entity_text.lower() != top_candidate.canonical_name.lower():
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
    def _compute_temporal_signals(self, entity_id: str):
        """
        Dynamically updates Recurrence, Unresolved State, and Sentiment trends.
        """
        entity = self.entities.get(entity_id)
        if not entity: return

        # A. Recurrence Score (Logarithmic scaling based on frequency)
        entity.recurrence_score = math.log10(1 + entity.mention_count)

        # B & C. Unresolved Score & Sentiment Trend
        event_ids = self.events_by_entity.get(entity_id, [])
        # Get actual event objects and sort by timestamp
        timeline_events = sorted([self.events[eid] for eid in event_ids], key=lambda x: x.timestamp)
        
        # Look at last 3 events
        recent_events = timeline_events[-3:]
        
        if recent_events:
            most_recent = recent_events[-1]
            if most_recent.event_type in ["problem", "risk"]:
                # Check if there is an update or decision resolving it recently
                if not any(e.event_type in ["decision", "update"] for e in recent_events):
                    entity.unresolved_score = 1.0
                else:
                    entity.unresolved_score = 0.4
            elif most_recent.event_type == "decision":
                entity.unresolved_score = 0.0
            else:
                entity.unresolved_score = 0.2

            # Sentiment trend (slope)
            sentiments = [e.sentiment for e in recent_events if e.sentiment != 0.0]
            if len(sentiments) > 1:
                # simple slope: last - first / length
                slope = (sentiments[-1] - sentiments[0]) / max(1, len(sentiments) - 1)
                entity.sentiment_trend = slope
            else:
                entity.sentiment_trend = 0.0

    def query_temporal_context(self, query_str: str) -> List[EventMemory]:
        """
        Layer 4 Reasoning Retrieval.
        Finds the closest entity to the query and returns all its historical events chronologically.
        """
        emb = self._get_embedding(query_str)
        if not emb:
            return []

        candidates = []
        for ent in self.entities.values():
            if ent.embedding:
                sim = self._cosine_similarity(emb, ent.embedding)
                candidates.append((ent, sim))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        if not candidates or candidates[0][1] < 0.75:
            return []

        best_entity = candidates[0][0]
        event_ids = self.events_by_entity.get(best_entity._id, [])
        return sorted([self.events[eid] for eid in event_ids], key=lambda x: x.timestamp)

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
        
    def ingest_meeting_results(self, meeting_id: str, scored_segments: List, importance_threshold: float = 0.4, clear: bool = False):
        """
        Batch ingest of scored SegmentFeatures into the temporal graph.
        Creates EntityMemory (topic nodes) and EventMemory (edges) from scored segments.
        Falls back to direct embedding-based canonicalization when LLM is not used.
        """
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

        for seg in scored_segments:
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

            # Get embedding (prefer pre-computed from SegmentFeatures)
            emb = emb_raw
            if emb is None and self.text_analyzer:
                arr = self.text_analyzer.get_embedding(text)
                emb = arr.tolist() if isinstance(arr, np.ndarray) else (arr or [])
            elif isinstance(emb, np.ndarray):
                emb = emb.tolist()
            elif emb is None:
                emb = []

            # Determine event type from scores
            if unresolved > 0.6:
                event_type = 'problem'
            elif semantic > 0.7:
                event_type = 'decision'
            else:
                event_type = 'discussion'

            # Canonicalize entity (topic) — use first 80 chars as entity name
            topic_label = text.strip()[:80]
            entity_node = self.resolve_entity(topic_label, 'topic')
            entity_node.speaker_stats[speaker] = entity_node.speaker_stats.get(speaker, 0) + 1
            entity_node.mention_count += 1
            entity_node.last_seen = datetime.now()

            # Create EventMemory record
            new_event = EventMemory(
                _id=self._generate_id(),
                entity_id=entity_node._id,
                meeting_id=meeting_id,
                timestamp=datetime.now(),
                speaker=speaker,
                event_type=event_type,
                summary=text.strip()[:200],
                sentiment=getattr(seg, 'tonal_score', 0.0) - 0.5,  # tonal → rough sentiment proxy
                confidence=fused_score
            )
            self.events[new_event._id] = new_event
            self.events_by_entity[entity_node._id].append(new_event._id)
            self.events_by_meeting[meeting_id].append(new_event._id)

            # Update temporal signals
            self._compute_temporal_signals(entity_node._id)

            ingested_events.append(new_event)
            if event_type == 'decision':
                ingested_decisions.append(new_event)
            elif event_type == 'problem':
                ingested_actions.append(new_event)
            else:
                ingested_topics.append(new_event)

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
        threads = []
        for ent in self.entities.values():
            mtgs = set(self.events[eid].meeting_id for eid in self.events_by_entity.get(ent._id, []))
            if len(mtgs) >= min_meetings:
                events = sorted([self.events[eid] for eid in self.events_by_entity.get(ent._id, [])], key=lambda x: x.timestamp)
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
        return threads

    def get_context_for_text(self, text: str, top_k: int = 5) -> List[Dict]:
        emb = self._get_embedding(text)
        if not emb: return []
        
        candidates = []
        for ent in self.entities.values():
            if ent.embedding:
                sim = self._cosine_similarity(emb, ent.embedding)
                candidates.append((ent, sim))
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        for ent, sim in candidates[:top_k]:
            last_ev = None
            evs = self.events_by_entity.get(ent._id, [])
            if evs:
                last_ev = self.events[evs[-1]]
            
            results.append({
                'type': ent.type,
                'similarity': sim,
                'timestamp': last_ev.timestamp.isoformat() if last_ev else ent.last_seen.isoformat(),
                'text': last_ev.summary if last_ev else ent.canonical_name,
                'name': ent.canonical_name,
                'speaker': last_ev.speaker if last_ev else 'Unknown'
            })
        return results

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
