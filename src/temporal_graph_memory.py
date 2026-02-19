"""
Temporal Graph Memory Module for Cross-Meeting Continuity.

Implements a graph-based memory system that:
1. Stores meeting segments with their embeddings and metadata
2. Links related topics, decisions, and action items across meetings
3. Enables temporal tracing of discussion threads
4. Provides context retrieval for the Highlight Scoring Model

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL GRAPH MEMORY                            │
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Meeting    │────▶│    Topic     │────▶│   Decision   │        │
│  │    Node      │     │    Node      │     │    Node      │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Segment    │────▶│  Action Item │────▶│   Follow-up  │        │
│  │    Node      │     │    Node      │     │    Node      │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│                                                                     │
│  Edge Types: CONTAINS, DISCUSSES, DECIDES, ASSIGNS, FOLLOWS_UP,    │
│              RELATES_TO, REFERENCES, CONTINUES_FROM                 │
└─────────────────────────────────────────────────────────────────────┘
"""

import json
import os
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the temporal graph."""
    MEETING = "meeting"
    SEGMENT = "segment"
    TOPIC = "topic"
    DECISION = "decision"
    ACTION_ITEM = "action_item"
    PERSON = "person"
    KEYWORD = "keyword"


class EdgeType(Enum):
    """Types of edges (relationships) in the temporal graph."""
    CONTAINS = "contains"           # Meeting contains segments
    DISCUSSES = "discusses"         # Segment discusses topic
    DECIDES = "decides"             # Segment makes a decision
    ASSIGNS = "assigns"             # Segment assigns action item
    FOLLOWS_UP = "follows_up"       # Action item follows up on previous
    RELATES_TO = "relates_to"       # Topic relates to another topic
    REFERENCES = "references"       # Segment references previous segment
    CONTINUES_FROM = "continues_from"  # Topic continues from previous meeting
    SPEAKER = "speaker"             # Segment has speaker
    MENTIONS = "mentions"           # Segment mentions keyword


@dataclass
class GraphNode:
    """A node in the temporal graph."""
    id: str
    type: NodeType
    content: str
    timestamp: datetime
    meeting_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # Store as list for JSON serialization
    importance_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'meeting_id': self.meeting_id,
            'metadata': self.metadata,
            'embedding': self.embedding,
            'importance_score': self.importance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GraphNode':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            type=NodeType(data['type']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            meeting_id=data.get('meeting_id'),
            metadata=data.get('metadata', {}),
            embedding=data.get('embedding'),
            importance_score=data.get('importance_score', 0.0)
        )


@dataclass
class GraphEdge:
    """An edge (relationship) in the temporal graph."""
    id: str
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.type.value,
            'weight': self.weight,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GraphEdge':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            type=EdgeType(data['type']),
            weight=data.get('weight', 1.0),
            metadata=data.get('metadata', {})
        )


class TemporalGraphMemory:
    """
    Temporal Graph Memory for cross-meeting continuity.
    
    Stores and retrieves contextual information across meeting sessions,
    enabling tracing of decisions and action items over time.
    """
    
    def __init__(
        self,
        storage_path: str = "meeting_memory",
        similarity_threshold: float = 0.75,
        max_context_nodes: int = 10,
        text_analyzer = None
    ):
        """
        Initialize the Temporal Graph Memory.
        
        Args:
            storage_path: Directory to store the graph data
            similarity_threshold: Minimum similarity for automatic linking
            max_context_nodes: Maximum nodes to return in context queries
            text_analyzer: Optional TextAnalyzer for generating embeddings
        """
        self.storage_path = storage_path
        self.similarity_threshold = similarity_threshold
        self.max_context_nodes = max_context_nodes
        self.text_analyzer = text_analyzer
        
        # Graph storage
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        
        # Indexes for fast lookup
        self.nodes_by_type: Dict[NodeType, Set[str]] = defaultdict(set)
        self.nodes_by_meeting: Dict[str, Set[str]] = defaultdict(set)
        self.edges_by_source: Dict[str, Set[str]] = defaultdict(set)
        self.edges_by_target: Dict[str, Set[str]] = defaultdict(set)
        self.edges_by_type: Dict[EdgeType, Set[str]] = defaultdict(set)
        
        # Topic tracking
        self.topic_keywords: Dict[str, Set[str]] = defaultdict(set)  # topic_id -> keywords
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing data
        self._load_graph()
        
        logger.info(f"TemporalGraphMemory initialized with {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using text analyzer if available."""
        if self.text_analyzer is not None:
            return self.text_analyzer.get_embedding(text)
        return None
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())[:8]
    
    # ==================== Node Operations ====================
    
    def add_node(self, node: GraphNode) -> str:
        """
        Add a node to the graph.
        
        Args:
            node: The node to add
            
        Returns:
            The node ID
        """
        if not node.id:
            node.id = self._generate_id()
            
        self.nodes[node.id] = node
        self.nodes_by_type[node.type].add(node.id)
        
        if node.meeting_id:
            self.nodes_by_meeting[node.meeting_id].add(node.id)
            
        logger.debug(f"Added node: {node.type.value} - {node.id}")
        return node.id
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        return [self.nodes[nid] for nid in self.nodes_by_type[node_type]]
    
    def get_nodes_by_meeting(self, meeting_id: str) -> List[GraphNode]:
        """Get all nodes from a specific meeting."""
        return [self.nodes[nid] for nid in self.nodes_by_meeting[meeting_id]]
    
    # ==================== Edge Operations ====================
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            weight: Edge weight (importance)
            metadata: Additional edge metadata
            
        Returns:
            The edge ID
        """
        edge_id = self._generate_id()
        edge = GraphEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            type=edge_type,
            weight=weight,
            metadata=metadata or {}
        )
        
        self.edges[edge_id] = edge
        self.edges_by_source[source_id].add(edge_id)
        self.edges_by_target[target_id].add(edge_id)
        self.edges_by_type[edge_type].add(edge_id)
        
        logger.debug(f"Added edge: {source_id} --{edge_type.value}--> {target_id}")
        return edge_id
    
    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Get all edges originating from a node."""
        return [self.edges[eid] for eid in self.edges_by_source[node_id]]
    
    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """Get all edges pointing to a node."""
        return [self.edges[eid] for eid in self.edges_by_target[node_id]]
    
    def get_connected_nodes(self, node_id: str) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get all nodes connected to a given node with their edges."""
        connected = []
        
        # Outgoing edges
        for edge_id in self.edges_by_source[node_id]:
            edge = self.edges[edge_id]
            target = self.nodes.get(edge.target_id)
            if target:
                connected.append((target, edge))
        
        # Incoming edges
        for edge_id in self.edges_by_target[node_id]:
            edge = self.edges[edge_id]
            source = self.nodes.get(edge.source_id)
            if source:
                connected.append((source, edge))
                
        return connected
    
    # ==================== Meeting Operations ====================
    
    def create_meeting(
        self,
        title: str,
        participants: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new meeting node.
        
        Args:
            title: Meeting title
            participants: List of participant names
            metadata: Additional meeting metadata
            
        Returns:
            Meeting node ID
        """
        meeting_id = self._generate_id()
        meeting_node = GraphNode(
            id=meeting_id,
            type=NodeType.MEETING,
            content=title,
            timestamp=datetime.now(),
            meeting_id=meeting_id,
            metadata={
                'participants': participants or [],
                **(metadata or {})
            }
        )
        
        self.add_node(meeting_node)
        
        # Create person nodes for participants
        for participant in (participants or []):
            person_id = self._get_or_create_person(participant)
            self.add_edge(meeting_id, person_id, EdgeType.CONTAINS)
        
        logger.info(f"Created meeting: {title} (ID: {meeting_id})")
        return meeting_id
    
    def _get_or_create_person(self, name: str) -> str:
        """Get existing person node or create new one."""
        # Search for existing person
        for node_id in self.nodes_by_type[NodeType.PERSON]:
            if self.nodes[node_id].content.lower() == name.lower():
                return node_id
        
        # Create new person node
        person_node = GraphNode(
            id=self._generate_id(),
            type=NodeType.PERSON,
            content=name,
            timestamp=datetime.now()
        )
        return self.add_node(person_node)
    
    def add_segment(
        self,
        meeting_id: str,
        text: str,
        start_time: float,
        end_time: float,
        speaker: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        importance_score: float = 0.0
    ) -> str:
        """
        Add a transcript segment to a meeting.
        
        Args:
            meeting_id: The meeting this segment belongs to
            text: Segment text content
            start_time: Start time in seconds
            end_time: End time in seconds
            speaker: Speaker name/ID
            embedding: Text embedding vector (auto-generated if text_analyzer available)
            importance_score: Computed importance score
            
        Returns:
            Segment node ID
        """
        # Auto-generate embedding if not provided and text_analyzer available
        if embedding is None and self.text_analyzer is not None:
            embedding = self._get_embedding(text)
        
        segment_node = GraphNode(
            id=self._generate_id(),
            type=NodeType.SEGMENT,
            content=text,
            timestamp=datetime.now(),
            meeting_id=meeting_id,
            metadata={
                'start_time': start_time,
                'end_time': end_time,
                'speaker': speaker
            },
            embedding=embedding.tolist() if embedding is not None else None,
            importance_score=importance_score
        )
        
        segment_id = self.add_node(segment_node)
        
        # Link to meeting
        self.add_edge(meeting_id, segment_id, EdgeType.CONTAINS)
        
        # Link to speaker if provided
        if speaker:
            person_id = self._get_or_create_person(speaker)
            self.add_edge(segment_id, person_id, EdgeType.SPEAKER)
        
        return segment_id
    
    # ==================== Topic & Decision Tracking ====================
    
    def add_topic(
        self,
        meeting_id: str,
        topic_text: str,
        keywords: Optional[List[str]] = None,
        segment_ids: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Add a topic discussed in a meeting.
        
        Args:
            meeting_id: The meeting where topic was discussed
            topic_text: Topic description
            keywords: Related keywords
            segment_ids: Segments where topic was discussed
            embedding: Topic embedding (auto-generated if text_analyzer available)
            
        Returns:
            Topic node ID
        """
        # Auto-generate embedding if not provided
        if embedding is None and self.text_analyzer is not None:
            embedding = self._get_embedding(topic_text)
        
        topic_node = GraphNode(
            id=self._generate_id(),
            type=NodeType.TOPIC,
            content=topic_text,
            timestamp=datetime.now(),
            meeting_id=meeting_id,
            metadata={'keywords': keywords or []},
            embedding=embedding.tolist() if embedding is not None else None
        )
        
        topic_id = self.add_node(topic_node)
        
        # Store keywords for lookup
        for kw in (keywords or []):
            self.topic_keywords[topic_id].add(kw.lower())
        
        # Link to meeting
        self.add_edge(meeting_id, topic_id, EdgeType.DISCUSSES)
        
        # Link to segments
        for seg_id in (segment_ids or []):
            self.add_edge(seg_id, topic_id, EdgeType.DISCUSSES)
        
        # Find and link related topics from previous meetings
        self._link_related_topics(topic_id, embedding)
        
        return topic_id
    
    def _link_related_topics(
        self,
        topic_id: str,
        embedding: Optional[np.ndarray] = None
    ):
        """Find and link related topics from previous meetings."""
        topic_node = self.nodes[topic_id]
        
        for other_id in self.nodes_by_type[NodeType.TOPIC]:
            if other_id == topic_id:
                continue
                
            other_node = self.nodes[other_id]
            
            # Don't link topics from the same meeting
            if other_node.meeting_id == topic_node.meeting_id:
                continue
            
            similarity = self._compute_similarity(topic_node, other_node)
            
            if similarity >= self.similarity_threshold:
                # Check if this is a continuation or just related
                time_diff = (topic_node.timestamp - other_node.timestamp).total_seconds()
                
                if time_diff > 0:  # Other topic is older
                    self.add_edge(
                        topic_id, other_id,
                        EdgeType.CONTINUES_FROM,
                        weight=similarity,
                        metadata={'similarity': similarity}
                    )
                else:
                    self.add_edge(
                        topic_id, other_id,
                        EdgeType.RELATES_TO,
                        weight=similarity,
                        metadata={'similarity': similarity}
                    )
                    
                logger.info(f"Linked topics: '{topic_node.content[:30]}...' <-> '{other_node.content[:30]}...' (sim={similarity:.2f})")
    
    def add_decision(
        self,
        meeting_id: str,
        decision_text: str,
        segment_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        participants: Optional[List[str]] = None
    ) -> str:
        """
        Record a decision made in a meeting.
        
        Args:
            meeting_id: The meeting where decision was made
            decision_text: The decision description
            segment_id: Segment where decision was stated
            topic_id: Related topic
            participants: People involved in decision
            
        Returns:
            Decision node ID
        """
        # Auto-generate embedding
        embedding = None
        if self.text_analyzer is not None:
            embedding = self._get_embedding(decision_text)
        
        decision_node = GraphNode(
            id=self._generate_id(),
            type=NodeType.DECISION,
            content=decision_text,
            timestamp=datetime.now(),
            meeting_id=meeting_id,
            metadata={'participants': participants or []},
            embedding=embedding.tolist() if embedding is not None else None
        )
        
        decision_id = self.add_node(decision_node)
        
        # Link to meeting
        self.add_edge(meeting_id, decision_id, EdgeType.DECIDES)
        
        # Link to segment
        if segment_id:
            self.add_edge(segment_id, decision_id, EdgeType.DECIDES)
        
        # Link to topic
        if topic_id:
            self.add_edge(topic_id, decision_id, EdgeType.DECIDES)
        
        logger.info(f"Recorded decision: {decision_text[:50]}...")
        return decision_id
    
    def add_action_item(
        self,
        meeting_id: str,
        action_text: str,
        assignee: Optional[str] = None,
        due_date: Optional[str] = None,
        segment_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        follows_up_on: Optional[str] = None,
        priority: str = "medium"
    ) -> str:
        """
        Record an action item from a meeting.
        
        Args:
            meeting_id: The meeting where action was assigned
            action_text: The action item description
            assignee: Person responsible
            due_date: Due date string
            segment_id: Segment where action was stated
            decision_id: Related decision
            follows_up_on: Previous action item this follows up on
            priority: Priority level (low, medium, high)
            
        Returns:
            Action item node ID
        """
        # Auto-generate embedding
        embedding = None
        if self.text_analyzer is not None:
            embedding = self._get_embedding(action_text)
        
        action_node = GraphNode(
            id=self._generate_id(),
            type=NodeType.ACTION_ITEM,
            content=action_text,
            timestamp=datetime.now(),
            meeting_id=meeting_id,
            metadata={
                'assignee': assignee,
                'due_date': due_date,
                'status': 'open',
                'priority': priority
            },
            embedding=embedding.tolist() if embedding is not None else None
        )
        
        action_id = self.add_node(action_node)
        
        # Link to meeting
        self.add_edge(meeting_id, action_id, EdgeType.ASSIGNS)
        
        # Link to segment
        if segment_id:
            self.add_edge(segment_id, action_id, EdgeType.ASSIGNS)
        
        # Link to decision
        if decision_id:
            self.add_edge(decision_id, action_id, EdgeType.ASSIGNS)
        
        # Link to assignee
        if assignee:
            person_id = self._get_or_create_person(assignee)
            self.add_edge(action_id, person_id, EdgeType.ASSIGNS)
        
        # Link to previous action item
        if follows_up_on:
            self.add_edge(action_id, follows_up_on, EdgeType.FOLLOWS_UP)
        
        logger.info(f"Recorded action item: {action_text[:50]}... (assigned to: {assignee})")
        return action_id
    
    def update_action_status(self, action_id: str, status: str):
        """Update the status of an action item."""
        if action_id in self.nodes:
            self.nodes[action_id].metadata['status'] = status
            logger.info(f"Updated action {action_id} status to: {status}")
    
    # ==================== Context Retrieval ====================
    
    def get_context_for_segment(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None,
        current_meeting_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get relevant context from previous meetings for a new segment.
        
        This is the main interface with the Highlight Scoring Model.
        
        Args:
            text: The segment text
            embedding: Text embedding vector
            current_meeting_id: Current meeting ID (to exclude)
            
        Returns:
            Dictionary containing:
            - related_topics: Previous related topics
            - related_decisions: Previous related decisions
            - open_action_items: Relevant open action items
            - discussion_history: Previous discussions on similar topics
            - context_score: Overall context relevance score
        """
        context = {
            'related_topics': [],
            'related_decisions': [],
            'open_action_items': [],
            'discussion_history': [],
            'context_score': 0.0,
            'cross_meeting_references': []
        }
        
        if embedding is None:
            return context
        
        # Create temporary node for similarity computation
        temp_node = GraphNode(
            id='temp',
            type=NodeType.SEGMENT,
            content=text,
            timestamp=datetime.now(),
            embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        )
        
        # Find related topics
        topic_scores = []
        for topic_id in self.nodes_by_type[NodeType.TOPIC]:
            topic_node = self.nodes[topic_id]
            if topic_node.meeting_id == current_meeting_id:
                continue
            
            similarity = self._compute_similarity(temp_node, topic_node)
            if similarity >= self.similarity_threshold * 0.8:  # Slightly lower threshold for context
                topic_scores.append((topic_node, similarity))
        
        # Sort by similarity and take top N
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        for topic_node, sim in topic_scores[:self.max_context_nodes]:
            context['related_topics'].append({
                'topic': topic_node.content,
                'meeting_id': topic_node.meeting_id,
                'meeting_date': topic_node.timestamp.isoformat(),
                'similarity': sim,
                'keywords': topic_node.metadata.get('keywords', [])
            })
            
            # Get decisions related to this topic
            for edge in self.get_edges_from(topic_node.id):
                if edge.type == EdgeType.DECIDES:
                    decision_node = self.nodes.get(edge.target_id)
                    if decision_node:
                        context['related_decisions'].append({
                            'decision': decision_node.content,
                            'meeting_id': decision_node.meeting_id,
                            'date': decision_node.timestamp.isoformat(),
                            'related_topic': topic_node.content
                        })
        
        # Find open action items that might be relevant
        for action_id in self.nodes_by_type[NodeType.ACTION_ITEM]:
            action_node = self.nodes[action_id]
            if action_node.metadata.get('status') != 'open':
                continue
            
            # Check keyword overlap or embedding similarity
            is_relevant = False
            action_lower = action_node.content.lower()
            text_lower = text.lower()
            
            # Simple keyword check
            keywords = text_lower.split()
            for kw in keywords:
                if len(kw) > 3 and kw in action_lower:
                    is_relevant = True
                    break
            
            if is_relevant:
                context['open_action_items'].append({
                    'action': action_node.content,
                    'assignee': action_node.metadata.get('assignee'),
                    'due_date': action_node.metadata.get('due_date'),
                    'meeting_id': action_node.meeting_id
                })
        
        # Find similar past segments (discussion history)
        segment_scores = []
        for seg_id in self.nodes_by_type[NodeType.SEGMENT]:
            seg_node = self.nodes[seg_id]
            if seg_node.meeting_id == current_meeting_id:
                continue
            
            similarity = self._compute_similarity(temp_node, seg_node)
            if similarity >= self.similarity_threshold:
                segment_scores.append((seg_node, similarity))
        
        segment_scores.sort(key=lambda x: x[1], reverse=True)
        for seg_node, sim in segment_scores[:5]:
            context['discussion_history'].append({
                'text': seg_node.content,
                'speaker': seg_node.metadata.get('speaker'),
                'meeting_id': seg_node.meeting_id,
                'date': seg_node.timestamp.isoformat(),
                'similarity': sim
            })
        
        # Compute overall context score
        if context['related_topics'] or context['related_decisions'] or context['open_action_items']:
            topic_weight = len(context['related_topics']) * 0.3
            decision_weight = len(context['related_decisions']) * 0.4
            action_weight = len(context['open_action_items']) * 0.3
            
            max_sim = max(
                [t['similarity'] for t in context['related_topics']] +
                [d.get('similarity', 0.5) for d in context['discussion_history']] +
                [0]
            )
            
            context['context_score'] = min(1.0, (topic_weight + decision_weight + action_weight) * max_sim)
            
            # Create cross-meeting references
            meetings_referenced = set()
            for topic in context['related_topics']:
                meetings_referenced.add(topic['meeting_id'])
            for decision in context['related_decisions']:
                meetings_referenced.add(decision['meeting_id'])
            
            for mid in meetings_referenced:
                meeting_node = self.nodes.get(mid)
                if meeting_node:
                    context['cross_meeting_references'].append({
                        'meeting_id': mid,
                        'meeting_title': meeting_node.content,
                        'date': meeting_node.timestamp.isoformat()
                    })
        
        return context
    
    def trace_topic_history(self, topic_id: str) -> List[Dict]:
        """
        Trace the history of a topic across meetings.
        
        Args:
            topic_id: The topic node ID
            
        Returns:
            Chronological list of topic discussions
        """
        history = []
        visited = set()
        
        def traverse(node_id: str, depth: int = 0):
            if node_id in visited or depth > 10:
                return
            visited.add(node_id)
            
            node = self.nodes.get(node_id)
            if not node:
                return
            
            history.append({
                'topic': node.content,
                'meeting_id': node.meeting_id,
                'date': node.timestamp.isoformat(),
                'depth': depth
            })
            
            # Follow CONTINUES_FROM and RELATES_TO edges
            for edge in self.get_edges_from(node_id):
                if edge.type in [EdgeType.CONTINUES_FROM, EdgeType.RELATES_TO]:
                    traverse(edge.target_id, depth + 1)
            
            for edge in self.get_edges_to(node_id):
                if edge.type in [EdgeType.CONTINUES_FROM, EdgeType.RELATES_TO]:
                    traverse(edge.source_id, depth + 1)
        
        traverse(topic_id)
        
        # Sort by date
        history.sort(key=lambda x: x['date'])
        return history
    
    def get_action_item_chain(self, action_id: str) -> List[Dict]:
        """
        Get the chain of follow-up action items.
        
        Args:
            action_id: Starting action item ID
            
        Returns:
            List of action items in the chain
        """
        chain = []
        visited = set()
        
        def traverse(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            node = self.nodes.get(node_id)
            if not node or node.type != NodeType.ACTION_ITEM:
                return
            
            chain.append({
                'action': node.content,
                'assignee': node.metadata.get('assignee'),
                'status': node.metadata.get('status'),
                'meeting_id': node.meeting_id,
                'date': node.timestamp.isoformat()
            })
            
            # Follow FOLLOWS_UP edges
            for edge in self.get_edges_from(node_id):
                if edge.type == EdgeType.FOLLOWS_UP:
                    traverse(edge.target_id)
            
            for edge in self.get_edges_to(node_id):
                if edge.type == EdgeType.FOLLOWS_UP:
                    traverse(edge.source_id)
        
        traverse(action_id)
        chain.sort(key=lambda x: x['date'])
        return chain
    
    # ==================== Similarity Computation ====================
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            return float(dot / (norm1 * norm2))
        return 0.0
    
    def _compute_similarity(self, node1: GraphNode, node2: GraphNode) -> float:
        """
        Compute similarity between two nodes.
        
        Uses embedding cosine similarity if available, falls back to keyword overlap.
        """
        # Try embedding similarity first
        if node1.embedding and node2.embedding:
            vec1 = np.array(node1.embedding)
            vec2 = np.array(node2.embedding)
            
            # Cosine similarity
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                return float(dot / (norm1 * norm2))
        
        # Fallback to keyword overlap
        # Only use words with 4+ characters to avoid trivial matches
        import re as _re
        words1 = set(_re.findall(r'\b[a-zA-Z]{4,}\b', node1.content.lower()))
        words2 = set(_re.findall(r'\b[a-zA-Z]{4,}\b', node2.content.lower()))
        
        # Remove common stop words AND filler words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'again', 'further', 'then', 'once',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                     'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                     'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'any', 'this', 'that', 'these', 'those',
                     'you', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                     # Extended filler/generic words that cause false matches
                     'that', 'this', 'with', 'have', 'they', 'from', 'some', 'been',
                     'also', 'just', 'then', 'than', 'into', 'about', 'over',
                     'your', 'ours', 'their', 'were', 'said', 'says', 'like',
                     'know', 'think', 'going', 'want', 'need', 'make', 'made',
                     'come', 'came', 'went', 'take', 'took', 'look', 'because',
                     'really', 'very', 'actually', 'basically', 'right', 'okay',
                     'yeah', 'well', 'good', 'great', 'kind', 'sort', 'thing',
                     'things', 'definitely', 'probably', 'maybe', 'perhaps',
                     'again', 'still', 'already', 'always', 'never', 'often',
                     'even', 'back', 'much', 'many', 'most', 'each', 'every',
                     'other', 'another', 'something', 'anything', 'nothing',
                     'everything', 'someone', 'anyone', 'hopefully', 'obviously',
                     'help', 'does', 'doesn', 'team', 'rest', 'effective',
                     'either', 'folks', 'people'}
        
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        # Require at least 2 overlapping words to count as similar
        if len(intersection) < 2:
            return 0.0
        
        return len(intersection) / len(union) if union else 0.0
    
    # ==================== Persistence ====================
    
    def save(self):
        """Save the graph to disk."""
        data = {
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': {eid: edge.to_dict() for eid, edge in self.edges.items()},
            'topic_keywords': {tid: list(kws) for tid, kws in self.topic_keywords.items()}
        }
        
        filepath = os.path.join(self.storage_path, 'graph_data.json')
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    def _load_graph(self):
        """Load the graph from disk."""
        filepath = os.path.join(self.storage_path, 'graph_data.json')
        
        if not os.path.exists(filepath):
            logger.info("No existing graph data found, starting fresh")
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load nodes
            for nid, node_data in data.get('nodes', {}).items():
                node = GraphNode.from_dict(node_data)
                self.nodes[nid] = node
                self.nodes_by_type[node.type].add(nid)
                if node.meeting_id:
                    self.nodes_by_meeting[node.meeting_id].add(nid)
            
            # Load edges
            for eid, edge_data in data.get('edges', {}).items():
                edge = GraphEdge.from_dict(edge_data)
                self.edges[eid] = edge
                self.edges_by_source[edge.source_id].add(eid)
                self.edges_by_target[edge.target_id].add(eid)
                self.edges_by_type[edge.type].add(eid)
            
            # Load topic keywords
            for tid, kws in data.get('topic_keywords', {}).items():
                self.topic_keywords[tid] = set(kws)
            
            logger.info(f"Loaded graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
    
    # ==================== Query Interface ====================
    
    def search_by_keyword(self, keyword: str, node_types: Optional[List[NodeType]] = None) -> List[GraphNode]:
        """
        Search nodes by keyword.
        
        Args:
            keyword: Keyword to search for
            node_types: Optional list of node types to search
            
        Returns:
            List of matching nodes
        """
        keyword_lower = keyword.lower()
        results = []
        
        types_to_search = node_types or list(NodeType)
        
        for node_type in types_to_search:
            for node_id in self.nodes_by_type[node_type]:
                node = self.nodes[node_id]
                if keyword_lower in node.content.lower():
                    results.append(node)
        
        return results
    
    def get_meeting_summary(self, meeting_id: str) -> Dict:
        """
        Get a summary of a meeting's graph data.
        
        Args:
            meeting_id: Meeting ID
            
        Returns:
            Summary dictionary
        """
        meeting_node = self.nodes.get(meeting_id)
        if not meeting_node:
            return {}
        
        summary = {
            'title': meeting_node.content,
            'date': meeting_node.timestamp.isoformat(),
            'participants': meeting_node.metadata.get('participants', []),
            'segments': [],
            'topics': [],
            'decisions': [],
            'action_items': []
        }
        
        for node_id in self.nodes_by_meeting[meeting_id]:
            node = self.nodes[node_id]
            
            if node.type == NodeType.SEGMENT:
                summary['segments'].append({
                    'text': node.content,
                    'speaker': node.metadata.get('speaker'),
                    'importance': node.importance_score
                })
            elif node.type == NodeType.TOPIC:
                summary['topics'].append({
                    'topic': node.content,
                    'keywords': node.metadata.get('keywords', [])
                })
            elif node.type == NodeType.DECISION:
                summary['decisions'].append({
                    'decision': node.content,
                    'participants': node.metadata.get('participants', [])
                })
            elif node.type == NodeType.ACTION_ITEM:
                summary['action_items'].append({
                    'action': node.content,
                    'assignee': node.metadata.get('assignee'),
                    'status': node.metadata.get('status')
                })
        
        return summary
    
    def get_all_meetings(self) -> List[GraphNode]:
        """Get all meeting nodes sorted by timestamp."""
        meeting_nodes = [
            self.nodes[node_id] 
            for node_id in self.nodes_by_type[NodeType.MEETING]
        ]
        return sorted(meeting_nodes, key=lambda x: x.timestamp, reverse=True)
    
    def get_context_for_text(
        self, 
        text: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context across all meetings using text similarity.
        
        Args:
            text: Search query text
            top_k: Maximum number of results to return
            
        Returns:
            List of relevant items with type, content, and similarity score
        """
        if self.text_analyzer is None:
            logger.warning("No text_analyzer available for semantic search")
            return []
        
        query_embedding = self._get_embedding(text)
        if query_embedding is None:
            return []
        
        results = []
        
        # Search through all nodes with embeddings
        for node in self.nodes.values():
            if node.embedding is None:
                continue
                
            node_embedding = np.array(node.embedding)
            similarity = self._cosine_similarity(query_embedding, node_embedding)
            
            if similarity > 0.3:  # Minimum threshold
                item = {
                    'node_id': node.id,
                    'type': node.type.value,
                    'similarity': float(similarity),
                    'timestamp': node.timestamp.isoformat() if node.timestamp else None,
                    'meeting_id': node.meeting_id
                }
                
                # Add type-specific fields
                if node.type == NodeType.SEGMENT:
                    item['text'] = node.content
                    item['speaker'] = node.metadata.get('speaker')
                    item['importance'] = node.importance_score
                elif node.type == NodeType.TOPIC:
                    item['name'] = node.content
                    item['keywords'] = node.metadata.get('keywords', [])
                elif node.type == NodeType.DECISION:
                    item['text'] = node.content
                    item['participants'] = node.metadata.get('participants', [])
                elif node.type == NodeType.ACTION_ITEM:
                    item['text'] = node.content
                    item['assignee'] = node.metadata.get('assignee')
                    item['status'] = node.metadata.get('status')
                    item['priority'] = node.metadata.get('priority', 'medium')
                
                results.append(item)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def get_temporal_boost_for_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[float, List[Dict]]:
        """
        Compute temporal context boost for highlight scoring.
        
        Args:
            embedding: Text embedding to match
            top_k: Maximum context items to consider
            
        Returns:
            Tuple of (boost_score, relevant_context_items)
        """
        relevant_context = []
        
        # Search for similar content in history
        for node in self.nodes.values():
            if node.embedding is None:
                continue
            
            # Skip segments (we want higher-level context)
            if node.type == NodeType.SEGMENT:
                continue
                
            node_embedding = np.array(node.embedding)
            similarity = self._cosine_similarity(embedding, node_embedding)
            
            if similarity > self.similarity_threshold:
                context_item = {
                    'type': node.type.value,
                    'content': node.content,
                    'similarity': float(similarity),
                    'node_id': node.id
                }
                
                # Add importance weight based on type
                if node.type == NodeType.DECISION:
                    context_item['weight'] = 1.5
                elif node.type == NodeType.ACTION_ITEM:
                    if node.metadata.get('status') == 'open':
                        context_item['weight'] = 1.3
                    else:
                        context_item['weight'] = 0.8
                elif node.type == NodeType.TOPIC:
                    context_item['weight'] = 1.0
                else:
                    context_item['weight'] = 0.5
                
                relevant_context.append(context_item)
        
        # Sort by weighted similarity
        relevant_context.sort(
            key=lambda x: x['similarity'] * x['weight'], 
            reverse=True
        )
        top_context = relevant_context[:top_k]
        
        # Compute boost score
        if not top_context:
            return 0.0, []
        
        # Average weighted similarity
        total_weight = sum(c['similarity'] * c['weight'] for c in top_context)
        boost = min(total_weight / top_k, 1.0)  # Normalize to 0-1
        
        return boost, top_context
    
    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'meetings': len(self.nodes_by_type[NodeType.MEETING]),
            'segments': len(self.nodes_by_type[NodeType.SEGMENT]),
            'topics': len(self.nodes_by_type[NodeType.TOPIC]),
            'decisions': len(self.nodes_by_type[NodeType.DECISION]),
            'action_items': len(self.nodes_by_type[NodeType.ACTION_ITEM]),
            'persons': len(self.nodes_by_type[NodeType.PERSON])
        }

    # ==================== Auto-Ingestion from Scored Segments ====================

    def ingest_meeting_results(
        self,
        meeting_id: str,
        scored_segments: List[Any],
        importance_threshold: float = 0.4,
        auto_extract: bool = True,
        clear: bool = False
    ) -> Dict[str, List[str]]:
        """
        Auto-populate the graph from a list of SegmentFeatures after fusion analysis.

        Args:
            meeting_id: Target meeting node ID
            scored_segments: List of SegmentFeatures or dicts
            importance_threshold: Minimum fused_score to attempt topic/decision extraction
            auto_extract: If True, run keyword extraction on important segments
            clear: If True, remove all existing segments/topics/decisions/action_items
                   for this meeting before ingesting (prevents stale data accumulation)

        Returns:
            Dict mapping 'segments', 'topics', 'decisions', 'action_items' to node ID lists.
        """
        import re

        # --- Optional: clear existing data for this meeting first ---
        if clear:
            remove_types = {NodeType.SEGMENT, NodeType.TOPIC, NodeType.DECISION, NodeType.ACTION_ITEM}
            nodes_to_remove = [
                nid for nid, n in list(self.nodes.items())
                if n.meeting_id == meeting_id and n.type in remove_types
            ]
            for nid in nodes_to_remove:
                # Remove associated edges
                for eid in list(self.edges_by_source.get(nid, set())):
                    edge = self.edges.pop(eid, None)
                    if edge:
                        self.edges_by_target.get(edge.target_id, set()).discard(eid)
                        self.edges_by_type.get(edge.type, set()).discard(eid)
                for eid in list(self.edges_by_target.get(nid, set())):
                    edge = self.edges.pop(eid, None)
                    if edge:
                        self.edges_by_source.get(edge.source_id, set()).discard(eid)
                        self.edges_by_type.get(edge.type, set()).discard(eid)
                self.edges_by_source.pop(nid, None)
                self.edges_by_target.pop(nid, None)
                node = self.nodes.pop(nid, None)
                if node:
                    self.nodes_by_type[node.type].discard(nid)
                    self.nodes_by_meeting[meeting_id].discard(nid)
            logger.info(f"Cleared {len(nodes_to_remove)} existing nodes for meeting {meeting_id}")

        created: Dict[str, List[str]] = {
            'segments': [], 'topics': [], 'decisions': [], 'action_items': []
        }

        # Patterns for extraction
        DECISION_PATTERNS = re.compile(
            r'\b(decided|approved|agreed|confirmed|going with|we will|we are going|'
            r'final decision|resolved|concluded|chosen|selected)\b',
            re.IGNORECASE
        )
        ACTION_PATTERNS = re.compile(
            r'\b(action item|follow.?up|will do|needs? to|should|must|please|'
            r'assigned to|take care|handle this|schedule|set up|create|fix|'
            r'update|send|review|check|verify|implement|deploy)\b',
            re.IGNORECASE
        )

        # Helper to normalise SegmentFeatures OR plain dict
        def _get(seg, attr, default=None):
            if isinstance(seg, dict):
                return seg.get(attr, default)
            return getattr(seg, attr, default)

        for seg in scored_segments:
            text = _get(seg, 'text', '')
            if not text or not text.strip():
                continue

            start_time  = _get(seg, 'start_time', 0.0) or _get(seg, 'start', 0.0)
            end_time    = _get(seg, 'end_time',   0.0) or _get(seg, 'end',   0.0)
            speaker     = _get(seg, 'speaker')
            fused_score = _get(seg, 'fused_score', 0.0)
            text_emb    = _get(seg, 'text_embedding')  # numpy array or None

            # Convert numpy embedding to list if needed
            emb_list = None
            if text_emb is not None:
                try:
                    emb_list = text_emb.tolist()
                except AttributeError:
                    emb_list = list(text_emb)

            seg_id = self.add_segment(
                meeting_id=meeting_id,
                text=text,
                start_time=start_time,
                end_time=end_time,
                speaker=speaker,
                importance_score=fused_score
            )
            # Manually attach saved embedding to the node
            if emb_list:
                self.nodes[seg_id].embedding = emb_list
            created['segments'].append(seg_id)

            if not auto_extract or fused_score < importance_threshold:
                continue

            # --- Extract keywords for topic (meaningful words only) ---
            stop = {
                'that', 'this', 'with', 'have', 'they', 'from', 'some', 'been',
                'will', 'what', 'when', 'there', 'more', 'also', 'just', 'then',
                'than', 'into', 'about', 'over', 'your', 'ours', 'their', 'were',
                'said', 'says', 'like', 'know', 'think', 'going', 'want', 'need',
                'make', 'made', 'come', 'came', 'went', 'take', 'took', 'look',
                'because', 'really', 'very', 'actually', 'basically', 'right',
                'okay', 'yeah', 'well', 'good', 'great', 'kind', 'sort',
                'definitely', 'probably', 'maybe', 'perhaps', 'again', 'still',
                'already', 'always', 'never', 'often', 'could', 'would', 'should',
                'might', 'must', 'shall', 'does', 'doing', 'done', 'even', 'back',
                'after', 'before', 'during', 'since', 'between', 'while', 'though',
                'through', 'without', 'within', 'around', 'along', 'across', 'away',
                'little', 'much', 'many', 'most', 'each', 'every', 'other', 'another',
                'something', 'anything', 'nothing', 'everything', 'someone', 'anyone',
                'want', 'partner', 'demand', 'steve', 'hopefully', 'obviously',
                'doesn', 'help', 'folks', 'people', 'team', 'rest', 'effective',
                'either', 'seven', 'carriages', 'texas',
            }

            # Topic vocabulary — domain words that indicate a substantive topic
            TOPIC_VOCAB = {
                'security', 'vulnerability', 'authentication', 'authorization', 'breach',
                'backend', 'frontend', 'database', 'server', 'infrastructure', 'deployment',
                'kubernetes', 'docker', 'cloud', 'microservice', 'pipeline', 'latency',
                'performance', 'scalability', 'monitoring', 'logging', 'cache',
                'migration', 'downtime', 'incident', 'outage', 'staging', 'production',
                'release', 'rollout', 'feature', 'sprint', 'milestone', 'roadmap',
                'architecture', 'refactor', 'technical', 'debugging', 'error', 'exception',
                'repository', 'dependency', 'version', 'upgrade',
                'budget', 'cost', 'revenue', 'funding', 'investment', 'profit', 'loss',
                'contract', 'vendor', 'partnership', 'customer', 'client', 'stakeholder',
                'deadline', 'timeline', 'priority', 'blocker', 'bottleneck', 'scope',
                'decision', 'approval', 'strategy', 'planning', 'initiative', 'proposal',
                'launch', 'rollback', 'delay', 'escalation', 'review', 'audit',
                'compliance', 'legal', 'regulation', 'policy', 'process', 'workflow',
                'hiring', 'onboarding', 'resource', 'capacity', 'workload', 'bandwidth',
                'feedback', 'design', 'research', 'prototype', 'requirement', 'specification',
            }

            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            vocab_keywords = [w for w in words if w in TOPIC_VOCAB]
            content_words = [w for w in words if w not in stop and w not in vocab_keywords]

            seen_kw: Set[str] = set()
            keywords = []
            for w in vocab_keywords + content_words:
                if w not in seen_kw:
                    seen_kw.add(w)
                    keywords.append(w)
            keywords = keywords[:8]

            # ── Filler / small-talk filter ──
            FILLER_PATTERNS = re.compile(
                r'\b(hopefully|obviously|basically|actually|you know|i mean|'
                r'kind of|sort of|i guess|i think so|sounds good|'
                r'good morning|good afternoon|hello everyone|hi everyone|'
                r'how are you|nice to meet|let me know|'
                r'carriages|texas)\b',
                re.IGNORECASE
            )
            text_lower = text.lower().strip()
            is_filler = bool(FILLER_PATTERNS.search(text_lower))

            # Gate: require TOPIC_VOCAB match to create a topic.
            # Without a vocab match we skip — pure content-word count
            # allows too much noise ("hopefully the rest of the team").
            has_strong_vocab = len(vocab_keywords) >= 1
            has_deep_vocab = len(vocab_keywords) >= 2

            create_topic = False
            if has_deep_vocab:
                # Two+ domain words → always create topic
                create_topic = True
            elif has_strong_vocab and not is_filler:
                # One domain word, but not filler → create topic
                create_topic = True
            elif not is_filler and len(content_words) >= 7 and fused_score >= 0.6:
                # No domain vocab but the segment is long, substantive, and high-scoring
                create_topic = True

            if create_topic:
                # Use the actual sentence as the topic label — clean timestamps, cap at 80 chars
                clean_text = re.sub(r'^\[\d+\.?\d*s\]\s*', '', text).strip()
                if len(clean_text) > 80:
                    topic_label = clean_text[:77].rsplit(' ', 1)[0] + '…'
                else:
                    topic_label = clean_text
                # Prepend primary vocab keyword as a category hint
                if vocab_keywords:
                    topic_label = vocab_keywords[0].capitalize() + ': ' + topic_label

                topic_id = self.add_topic(
                    meeting_id=meeting_id,
                    topic_text=topic_label,
                    keywords=keywords,
                    segment_ids=[seg_id]
                )
                created['topics'].append(topic_id)

            # --- Extract decisions ---
            if DECISION_PATTERNS.search(text):
                decision_text = text[:160].strip()
                dec_id = self.add_decision(
                    meeting_id=meeting_id,
                    decision_text=decision_text,
                    segment_id=seg_id,
                    participants=[speaker] if speaker else None
                )
                created['decisions'].append(dec_id)

            # --- Extract action items ---
            if ACTION_PATTERNS.search(text):
                action_text = text[:160].strip()
                mention = re.search(r'@(\w+)', text)
                assignee = mention.group(1) if mention else speaker
                act_id = self.add_action_item(
                    meeting_id=meeting_id,
                    action_text=action_text,
                    assignee=assignee,
                    segment_id=seg_id
                )
                created['action_items'].append(act_id)

        logger.info(
            f"Ingested meeting {meeting_id}: "
            f"{len(created['segments'])} segments, "
            f"{len(created['topics'])} topics, "
            f"{len(created['decisions'])} decisions, "
            f"{len(created['action_items'])} action items"
        )
        return created

    # ==================== Cross-Meeting Thread Discovery ====================

    def find_cross_meeting_threads(
        self,
        min_meetings: int = 2,
        similarity_threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """
        Cluster topics that appear across >= min_meetings into recurring 'threads'.

        Algorithm:
        1. Fetch all TOPIC nodes.
        2. Build pairwise similarity (embedding cosine if available, else keyword Jaccard).
        3. Union-Find greedy clustering: merge nodes with sim >= threshold.
        4. Filter clusters spanning >= min_meetings unique meetings.
        5. Sort thread appearances chronologically.

        Returns:
            List of thread dicts with thread_id, label, first_seen, last_seen,
            meeting_count, keywords, appearances.
        """
        topic_ids = list(self.nodes_by_type[NodeType.TOPIC])
        if len(topic_ids) < 2:
            return []

        # --- Union-Find helper ---
        parent = {tid: tid for tid in topic_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # --- Cluster by pairwise similarity ---
        for i, tid_a in enumerate(topic_ids):
            node_a = self.nodes[tid_a]
            for tid_b in topic_ids[i + 1:]:
                node_b = self.nodes[tid_b]
                # Skip topics from the same meeting
                if node_a.meeting_id == node_b.meeting_id:
                    continue
                sim = self._compute_similarity(node_a, node_b)
                if sim >= similarity_threshold:
                    union(tid_a, tid_b)

        # --- Group into clusters ---
        clusters: Dict[str, List[str]] = defaultdict(list)
        for tid in topic_ids:
            clusters[find(tid)].append(tid)

        # Domain vocab for scoring labels
        _TOPIC_VOCAB = {
            'security', 'vulnerability', 'authentication', 'authorization', 'breach',
            'backend', 'frontend', 'database', 'server', 'infrastructure', 'deployment',
            'kubernetes', 'docker', 'cloud', 'microservice', 'pipeline', 'latency',
            'performance', 'scalability', 'monitoring', 'logging', 'cache',
            'migration', 'downtime', 'incident', 'outage', 'staging', 'production',
            'release', 'rollout', 'feature', 'sprint', 'milestone', 'roadmap',
            'architecture', 'refactor', 'technical', 'debugging', 'error', 'exception',
            'budget', 'cost', 'revenue', 'deadline', 'timeline', 'priority',
            'decision', 'approval', 'strategy', 'planning', 'design', 'research',
        }

        def _vocab_score(n):
            kws = n.metadata.get('keywords', [])
            return sum(1 for k in kws if k in _TOPIC_VOCAB)

        threads = []
        for root, members in clusters.items():
            # Collect unique meeting IDs in this cluster
            meeting_ids_in_cluster = {
                self.nodes[tid].meeting_id for tid in members
                if self.nodes[tid].meeting_id
            }
            if len(meeting_ids_in_cluster) < min_meetings:
                continue

            # Build appearances list (one per topic node, sorted by date)
            appearances = []
            all_keywords: Set[str] = set()
            for tid in members:
                node = self.nodes[tid]
                meeting_node = self.nodes.get(node.meeting_id)
                kws = node.metadata.get('keywords', [])
                all_keywords.update(kws)
                appearances.append({
                    'meeting_id': node.meeting_id,
                    'meeting_title': meeting_node.content if meeting_node else 'Unknown',
                    'date': node.timestamp.isoformat(),
                    'topic': node.content,
                    'keywords': kws
                })

            appearances.sort(key=lambda x: x['date'])

            # Label: pick the topic with the most domain-vocab keywords
            label_node = max(
                (self.nodes[tid] for tid in members),
                key=lambda n: (_vocab_score(n), len(n.metadata.get('keywords', [])))
            )

            threads.append({
                'thread_id': root,
                'label': label_node.content,
                'first_seen': appearances[0]['date'],
                'last_seen': appearances[-1]['date'],
                'meeting_count': len(meeting_ids_in_cluster),
                'keywords': sorted(all_keywords)[:10],
                'appearances': appearances
            })

        # Sort threads by most recently active first
        threads.sort(key=lambda t: t['last_seen'], reverse=True)
        logger.info(f"Found {len(threads)} cross-meeting threads")
        return threads

    def get_open_action_items(self) -> List[Dict[str, Any]]:
        """
        Return all open action items across all meetings, sorted by meeting date.
        """
        items = []
        for aid in self.nodes_by_type[NodeType.ACTION_ITEM]:
            node = self.nodes[aid]
            if node.metadata.get('status', 'open') != 'done':
                meeting_node = self.nodes.get(node.meeting_id)
                items.append({
                    'action_id': aid,
                    'action': node.content,
                    'assignee': node.metadata.get('assignee'),
                    'priority': node.metadata.get('priority', 'medium'),
                    'status': node.metadata.get('status', 'open'),
                    'meeting_id': node.meeting_id,
                    'meeting_title': meeting_node.content if meeting_node else 'Unknown',
                    'date': node.timestamp.isoformat()
                })
        items.sort(key=lambda x: x['date'])
        return items
