"""
Fusion Layer for Multi-Modal Meeting Analysis.

Combines four modalities to compute hyper-relevant importance scores:
1. Semantic (Text): BERT sentence embeddings capturing meaning
2. Tonal (Audio): MFCC/prosodic features capturing urgency/emphasis
3. Role: Static role embeddings for role-specific relevance
4. Temporal Context: Cross-meeting continuity from Temporal Graph Memory

The fusion produces a single relevance score per segment that reflects
both WHAT was said (semantic), HOW it was said (tonal), WHO it matters to (role),
and WHEN/WHERE it was discussed before (temporal context).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .ml_fusion import load_model, ContextualFusionTransformer
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML Fusion module not found. Falling back to heuristics.")

# Try to import Temporal Graph Memory
try:
    from .temporal_graph_memory import TemporalGraphMemory, NodeType
    TEMPORAL_MEMORY_AVAILABLE = True
except ImportError:
    TEMPORAL_MEMORY_AVAILABLE = False
    logger.warning("Temporal Graph Memory not available.")

# Try to import ParticipantStore
try:
    from .participant_store import ParticipantStore
    PARTICIPANT_STORE_AVAILABLE = True
except ImportError:
    PARTICIPANT_STORE_AVAILABLE = False
    logger.warning("ParticipantStore not available.")


@dataclass
class SegmentFeatures:
    """Container for multi-modal features of a transcript segment."""
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None
    
    # Embeddings (populated by analyzers)
    text_embedding: Optional[np.ndarray] = None  # 384-dim from MiniLM
    mfcc_embedding: Optional[np.ndarray] = None  # 52-dim from AudioTonalAnalyzer
    prosodic_features: Optional[Dict[str, float]] = None  # urgency, emphasis, etc.
    
    # Computed scores
    semantic_score: float = 0.0
    tonal_score: float = 0.0
    role_relevance: float = 0.0
    temporal_context_score: float = 0.0  # Cross-meeting context relevance
    recurrence_score: float = 0.0
    unresolved_score: float = 0.0
    fused_score: float = 0.0
    
    # Temporal context
    temporal_context: Optional[Dict[str, Any]] = None  # Context from previous meetings
    thread_info: Optional[Dict[str, Any]] = None       # Thread detection result for UI badges

    # Visual analysis
    is_screen_sharing: bool = False
    screen_share_confidence: float = 0.0
    ocr_text: str = ""
    visual_embedding: Optional[np.ndarray] = None


class FusionLayer:
    """
    Multi-modal fusion layer combining text, audio, and role signals.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      FUSION LAYER                           │
    │                                                             │
    │  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
    │  │  Text    │   │  Audio   │   │  Role    │                │
    │  │ Encoder  │   │ Encoder  │   │ Encoder  │                │
    │  │ (384-d)  │   │ (52-d)   │   │ (384-d)  │                │
    │  └────┬─────┘   └────┬─────┘   └────┬─────┘                │
    │       │              │              │                       │
    │       ▼              ▼              ▼                       │
    │  ┌─────────┐   ┌──────────┐   ┌──────────┐                 │
    │  │Semantic │   │  Tonal   │   │  Role    │                 │
    │  │ Score   │   │  Score   │   │Relevance │                 │
    │  │ (0-1)   │   │  (0-1)   │   │  (0-1)   │                 │
    │  └────┬────┘   └────┬─────┘   └────┬─────┘                 │
    │       │             │              │                        │
    │       └─────────────┼──────────────┘                        │
    │                     ▼                                       │
    │              ┌────────────┐                                 │
    │              │  Weighted  │                                 │
    │              │   Fusion   │                                 │
    │              │  Function  │                                 │
    │              └─────┬──────┘                                 │
    │                    ▼                                        │
    │              ┌────────────┐                                 │
    │              │   Fused    │                                 │
    │              │   Score    │                                 │
    │              │   (0-1)    │                                 │
    │              └────────────┘                                 │
    └─────────────────────────────────────────────────────────────┘
    
    Fusion Strategies:
    1. Weighted Sum: α*semantic + β*tonal + γ*role + δ*temporal
    2. Multiplicative: semantic * (1 + tonal_boost) * role_relevance * (1 + temporal_boost)
    3. Attention-based: Learn weights from cross-modal attention
    """
    
    def __init__(
        self,
        text_analyzer=None,
        audio_analyzer=None,
        temporal_memory: Optional['TemporalGraphMemory'] = None,
        participant_store: Optional[Any] = None,
        fusion_strategy: str = "weighted",
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the fusion layer.
        
        Args:
            text_analyzer: TextAnalyzer instance for semantic embeddings
            audio_analyzer: AudioTonalAnalyzer instance for prosodic features
            temporal_memory: TemporalGraphMemory instance for cross-meeting context
            fusion_strategy: "weighted", "multiplicative", or "gated"
            weights: Custom weights for fusion (default: balanced)
        """
        self.text_analyzer = text_analyzer
        self.audio_analyzer = audio_analyzer
        self.temporal_memory = temporal_memory
        self.participant_store = participant_store
        self.fusion_strategy = fusion_strategy
        
        # Current meeting ID (set when processing a meeting)
        self.current_meeting_id: Optional[str] = None
        
        # Thread detector (lazily initialised when temporal_memory is set)
        self._thread_detector = None
        if temporal_memory is not None and TEMPORAL_MEMORY_AVAILABLE:
            try:
                from .thread_detector import ThreadDetector
                self._thread_detector = ThreadDetector(temporal_memory)
                logger.info("ThreadDetector initialised inside FusionLayer")
            except ImportError:
                logger.warning("thread_detector module not found — thread boost disabled")
        
        # Default weights (can be tuned) - now includes explicit temporal signals
        self.weights = weights or {
            'semantic': 0.35,    # What was said
            'tonal': 0.15,      # How it was said
            'role': 0.20,       # Who it matters to
            'temporal': 0.10,    # General cross-meeting references
            'recurrence': 0.10,  # Specific recurrence signal
            'unresolved': 0.10   # Unresolved state signal
        }
        
        # Importance description embedding (cached)
        self._importance_embedding = None
        
        # Role embeddings cache
        self.role_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info(f"FusionLayer initialized with strategy: {fusion_strategy}")
        logger.info(f"Temporal Memory: {'enabled' if temporal_memory else 'disabled'}")
        logger.info(f"ParticipantStore: {'enabled' if participant_store else 'disabled'}")
        
        # ML Model
        self.ml_model = None
        if ML_AVAILABLE:
            try:
                self.ml_model = load_model("fusion_model.pth")
                logger.info("Loaded ML Fusion Model from fusion_model.pth")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}. Using heuristics.")
    
    def set_role_embeddings(self, role_embeddings: Dict[str, np.ndarray]):
        """Set pre-computed role embeddings."""
        self.role_embeddings = role_embeddings
        logger.info(f"Loaded {len(role_embeddings)} role embeddings")

    def update_fusion_weights(self, new_weights: Dict[str, float]):
        """Update fusion weights (e.g., from online learning feedback)."""
        self.weights = new_weights
        logger.info(f"Updated fusion weights: {new_weights}")
    
    def _get_importance_embedding(self, focus_query: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get cached embedding for 'important content' description or custom focus.
        
        Args:
            focus_query: Optional custom focus query (e.g., "budget concerns")
            
        Returns:
            Embedding vector
        """
        if not self.text_analyzer:
            return None
            
        # If custom focus is provided, generate embedding on the fly (don't cache globally)
        if focus_query and focus_query.strip():
            return self.text_analyzer.get_embedding(focus_query)
        
        # Default importance embedding (cached)
        if self._importance_embedding is not None:
            return self._importance_embedding
        
        importance_desc = """
        Important business decisions, strategic proposals, key insights,
        data-driven observations, problem identification, solution suggestions,
        action items, concerns raised, metrics discussion, process improvements,
        technical explanations, resource allocation, timeline commitments,
        risk assessment, and substantive professional contributions.
        """
        
        self._importance_embedding = self.text_analyzer.get_embedding(importance_desc)
        return self._importance_embedding
    
    def compute_semantic_score(
        self,
        text: str,
        text_embedding: Optional[np.ndarray] = None,
        focus_query: Optional[str] = None
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute semantic importance score for text.
        
        Args:
            text: The text content
            text_embedding: Pre-computed embedding (optional)
            focus_query: Optional custom focus query
            
        Returns:
            Tuple of (score, embedding)
        """
        if not self.text_analyzer:
            return 0.0, None
        
        # Get text embedding
        if text_embedding is None:
            text_embedding = self.text_analyzer.get_embedding(text)
        
        if text_embedding is None:
            return 0.0, None
        
        # Compare to importance description or focus query
        importance_emb = self._get_importance_embedding(focus_query)
        if importance_emb is None:
            return 0.0, text_embedding
        
        # Cosine similarity
        similarity = self._cosine_similarity(text_embedding, importance_emb)
        
        # Normalize to 0-1 (similarity can be negative)
        score = max(0.0, (similarity + 1) / 2)
        
        return float(score), text_embedding
    
    def compute_tonal_score(
        self,
        prosodic_features: Optional[Dict[str, float]] = None,
        mfcc_embedding: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute tonal importance score from prosodic features.
        
        High urgency + high emphasis = high tonal score
        
        Args:
            prosodic_features: Dict with urgency_score, emphasis_score, etc.
            mfcc_embedding: MFCC embedding (for future use)
            
        Returns:
            Tonal score (0-1)
        """
        if prosodic_features is None:
            return 0.0
        
        urgency = prosodic_features.get('urgency_score', 0.0)
        emphasis = prosodic_features.get('emphasis_score', 0.0)
        energy_std = prosodic_features.get('energy_std', 0.0)
        
        # Combine signals
        # High variation in energy suggests dynamic speech (more engaging)
        energy_factor = min(energy_std * 20, 1.0)  # Normalize
        
        # Weighted combination
        tonal_score = (
            0.4 * urgency +
            0.4 * emphasis +
            0.2 * energy_factor
        )
        
        return float(min(max(tonal_score, 0.0), 1.0))
    
    def compute_role_relevance(
        self,
        text_embedding: np.ndarray,
        role: str
    ) -> float:
        """
        Compute relevance of content to a specific role.
        
        Args:
            text_embedding: Text embedding vector
            role: Role name (e.g., "Product Manager")
            
        Returns:
            Role relevance score (0-1)
        """
        if text_embedding is None:
            return 0.0
        
        # Get role embedding
        role_emb = self.role_embeddings.get(role)
        
        if role_emb is None and self.text_analyzer:
            # Generate on-the-fly
            role_emb = self.text_analyzer.get_embedding(role)
            if role_emb is not None:
                self.role_embeddings[role] = role_emb
        
        if role_emb is None:
            return 0.5  # Neutral if no role embedding
        
        # Cosine similarity
        similarity = self._cosine_similarity(text_embedding, role_emb)
        
        # Normalize to 0-1
        score = max(0.0, (similarity + 1) / 2)
        
        return float(score)
    
    def compute_temporal_context_score(
        self,
        text: str,
        text_embedding: Optional[np.ndarray] = None
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Compute temporal context score from cross-meeting memory.
        
        Higher scores for segments that relate to previous discussions,
        decisions, or open action items.
        
        Args:
            text: Segment text
            text_embedding: Pre-computed text embedding
            
        Returns:
            Tuple of (temporal_score, context_dict)
        """
        if not self.temporal_memory or not TEMPORAL_MEMORY_AVAILABLE:
            return 0.0, None
        
        # Get context from temporal memory
        context = self.temporal_memory.get_context_for_segment(
            text=text,
            embedding=text_embedding,
            current_meeting_id=self.current_meeting_id
        )
        
        # The context_score is already computed by the memory
        temporal_score = context.get('context_score', 0.0)
        
        # Boost score if there are open action items related to this segment
        if context.get('open_action_items'):
            temporal_score = min(1.0, temporal_score + 0.2)
        
        # Boost score if this continues a previous topic
        if context.get('related_topics'):
            temporal_score = min(1.0, temporal_score + 0.1)
        
        return float(temporal_score), context
    
    def fuse_scores(
        self,
        semantic_score: float,
        tonal_score: float,
        role_relevance: float,
        temporal_score: float = 0.0,
        recurrence_score: float = 0.0,
        unresolved_score: float = 0.0
    ) -> float:
        """Fuse using the layer-level global weights. Delegates to _fuse_with_weights."""
        return self._fuse_with_weights(
            semantic_score, tonal_score, role_relevance, temporal_score, recurrence_score, unresolved_score, self.weights
        )

    def _fuse_with_weights(
        self,
        semantic_score: float,
        tonal_score: float,
        role_relevance: float,
        temporal_score: float,
        recurrence_score: float,
        unresolved_score: float,
        weights: Dict[str, float]
    ) -> float:
        """
        Fuse multi-modal scores into a single relevance score using explicit weights.

        This is the real implementation. fuse_scores() and score_segment() both
        delegate here — score_segment may supply per-speaker weights from
        ParticipantStore instead of the global self.weights.

        Args:
            semantic_score: Semantic importance (0-1)
            tonal_score: Tonal/prosodic importance (0-1)
            role_relevance: Role-specific relevance (0-1)
            temporal_score: Cross-meeting context relevance (0-1)
            recurrence_score: Dynamic recurrence tracked across graphs
            unresolved_score: High if item is a problem left open
            weights: Weight dict

        Returns:
            Fused score (0-1)
        """
        if self.fusion_strategy == "weighted":
            fused = (
                weights.get('semantic', 0.35) * semantic_score +
                weights.get('tonal', 0.15) * tonal_score +
                weights.get('role', 0.20) * role_relevance +
                weights.get('temporal', 0.10) * temporal_score +
                weights.get('recurrence', 0.10) * recurrence_score +
                weights.get('unresolved', 0.10) * unresolved_score
            )

        elif self.fusion_strategy == "multiplicative":
            tonal_boost = 1.0 + (tonal_score * 0.5)
            temporal_boost = 1.0 + (temporal_score * 0.3) + (recurrence_score * 0.2) + (unresolved_score * 0.2)
            fused = semantic_score * tonal_boost * temporal_boost * (0.5 + role_relevance * 0.5)

        elif self.fusion_strategy == "gated":
            gate = self._sigmoid((role_relevance - 0.5) * 4)
            content_score = (
                weights.get('semantic', 0.5) * semantic_score +
                weights.get('tonal', 0.2) * tonal_score +
                weights.get('temporal', 0.1) * temporal_score +
                weights.get('recurrence', 0.1) * recurrence_score +
                weights.get('unresolved', 0.1) * unresolved_score
            )
            fused = content_score * gate

        else:
            fused = (semantic_score + tonal_score + role_relevance) / 3

        return float(min(max(fused, 0.0), 1.0))

    
    def score_segment(
        self,
        segment: SegmentFeatures,
        role: str,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        focus_query: Optional[str] = None
    ) -> SegmentFeatures:
        """
        Score a single segment using all modalities.
        
        Args:
            segment: SegmentFeatures with text and timing
            role: Target role for relevance scoring
            audio_data: Full audio array (for extracting segment audio)
            sample_rate: Audio sample rate
            focus_query: Optional custom focus query
            
        Returns:
            SegmentFeatures with all scores populated
        """
        # 0. Resolve per-speaker weights from ParticipantStore (if available)
        active_weights = self.weights
        if self.participant_store and segment.speaker:
            speaker_weights = self.participant_store.get_weights_for_speaker(segment.speaker)
            if speaker_weights:
                active_weights = speaker_weights
                logger.debug(
                    f"Using profile weights for speaker '{segment.speaker}': {speaker_weights}"
                )

        # 1. Semantic score
        semantic_score, text_emb = self.compute_semantic_score(
            segment.text,
            segment.text_embedding,
            focus_query
        )
        segment.semantic_score = semantic_score
        segment.text_embedding = text_emb
        
        # 2. Tonal score (if audio available)
        if audio_data is not None and self.audio_analyzer:
            # Extract segment audio
            start_sample = int(segment.start_time * sample_rate)
            end_sample = int(segment.end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) > 0:
                segment.prosodic_features = self.audio_analyzer.extract_prosodic_features(
                    segment_audio
                )
                segment.mfcc_embedding = self.audio_analyzer.get_mfcc_embedding(
                    segment_audio
                )
        
        tonal_score = self.compute_tonal_score(
            segment.prosodic_features,
            segment.mfcc_embedding
        )
        segment.tonal_score = tonal_score
        
        # 3. Role relevance
        if text_emb is not None:
            role_relevance = self.compute_role_relevance(text_emb, role)
        else:
            role_relevance = 0.5
        segment.role_relevance = role_relevance
        
        # 4. Temporal context (cross-meeting relevance)
        temporal_score = 0.0
        recurrence_score = 0.0
        unresolved_score = 0.0
        
        if text_emb is not None:
            temporal_score, temporal_context = self.compute_temporal_context_score(
                text=segment.text, text_embedding=text_emb
            )
            segment.temporal_context_score = temporal_score
            segment.temporal_context = temporal_context
            
            # Extract precise signals matching our 4-layer architecture graph output
            if temporal_context and 'signals' in temporal_context:
                sig_dict = temporal_context['signals']
                recurrence_score = sig_dict.get('recurrence_score', 0.0)
                unresolved_score = sig_dict.get('unresolved_score', 0.0)
            
            segment.recurrence_score = recurrence_score
            segment.unresolved_score = unresolved_score

        # 4b. Thread boost — extra weight if segment belongs to a recurring thread
        if self._thread_detector is not None and text_emb is not None:
            try:
                thread = self._thread_detector.get_thread_for_segment(
                    segment.text, text_emb
                )
                if thread:
                    days_since = thread.days_since_first()
                    # Boost scales with recency: older threads still get 30 % base boost
                    recency_factor = max(0.3, 1.0 - days_since / 30.0)
                    thread_boost = 0.25 * recency_factor * min(thread.meeting_count, 4) / 4
                    temporal_score = min(1.0, temporal_score + thread_boost)
                    segment.temporal_context_score = temporal_score
                    segment.thread_info = {
                        'thread_id': thread.thread_id,
                        'label': thread.canonical_label,
                        'first_seen': thread.first_seen,
                        'meeting_count': thread.meeting_count,
                        'annotation': self._thread_detector.generate_thread_annotation(thread),
                        'keywords': thread.keywords[:5],
                    }
            except Exception as e:
                logger.debug(f"Thread detection skipped: {e}")
        
        # 5. Fuse scores (use speaker-specific weights if resolved)
        segment.fused_score = self._fuse_with_weights(
            semantic_score, tonal_score, role_relevance, temporal_score, recurrence_score, unresolved_score, active_weights
        )
        
        return segment
    
    def score_segments(
        self,
        segments: List[Dict],
        role: str,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        focus_query: Optional[str] = None
    ) -> List[SegmentFeatures]:
        """
        Score multiple segments.
        
        Args:
            segments: List of dicts with 'text', 'start', 'end', 'speaker'
            role: Target role
            audio_data: Full audio array
            sample_rate: Audio sample rate
            focus_query: Optional custom focus query
            
        Returns:
            List of scored SegmentFeatures
        """
        scored = []
        
        for seg in segments:
            features = SegmentFeatures(
                start_time=seg.get('start', 0),
                end_time=seg.get('end', 0),
                text=seg.get('text', ''),
                speaker=seg.get('speaker')
            )
            
            scored_segment = self.score_segment(
                features,
                role,
                audio_data,
                sample_rate,
                focus_query
            )
            scored.append(scored_segment)
        
        return scored

    def score_segments_contextual(
        self,
        segments: List[Dict],
        role: str,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        focus_query: Optional[str] = None,
        use_ml: bool = True
    ) -> List[SegmentFeatures]:
        """
        Score segments using Contextual ML Model if available, else Heuristics.
        """
        # First, populate features (embeddings, etc.) using the standard scoring logic
        # We need the embeddings for the ML model
        scored_segments = self.score_segments(segments, role, audio_data, sample_rate, focus_query)
        
        if use_ml and self.ml_model and ML_AVAILABLE:
            try:
                import torch
                
                # Prepare Batch
                # 1. Text
                text_embs = [s.text_embedding if s.text_embedding is not None else np.zeros(384) for s in scored_segments]
                text_tensor = torch.tensor(np.array(text_embs), dtype=torch.float32).unsqueeze(0) # (1, Seq, 384)
                
                # 2. Audio
                audio_embs = []
                for s in scored_segments:
                    if s.mfcc_embedding is not None:
                        emb = s.mfcc_embedding
                        if len(emb) < 64:
                            emb = np.pad(emb, (0, 64 - len(emb)))
                        elif len(emb) > 64:
                            emb = emb[:64]
                        audio_embs.append(emb)
                    else:
                        audio_embs.append(np.zeros(64))
                audio_tensor = torch.tensor(np.array(audio_embs), dtype=torch.float32).unsqueeze(0)
                
                # 3. Role
                role_embs = []
                # We need role embedding vector. compute_role_relevance uses it but doesn't store it in segment
                # Let's fetch it from cache
                role_vec = self.role_embeddings.get(role)
                if role_vec is None and self.text_analyzer:
                    role_vec = self.text_analyzer.get_embedding(role)
                
                if role_vec is None:
                    role_vec = np.zeros(384)
                    
                # Repeat role vector for all segments (since role is constant for the query)
                role_tensor = torch.tensor(np.array([role_vec] * len(scored_segments)), dtype=torch.float32).unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    scores = self.ml_model(text_tensor, audio_tensor, role_tensor)
                    # scores: (1, Seq, 1)
                    
                # Assign scores
                ml_scores = scores.squeeze().numpy()
                if ml_scores.ndim == 0:
                    ml_scores = [float(ml_scores)]
                    
                for i, seg in enumerate(scored_segments):
                    seg.fused_score = float(ml_scores[i])
                    # We keep the component scores (semantic, tonal, role) for explanation,
                    # but override the fused_score with the ML prediction.
                    
                logger.info("Scored segments using ML Fusion Model")
                return scored_segments
                
            except Exception as e:
                logger.error(f"ML Scoring failed: {e}. Falling back to heuristics.")
                return scored_segments
        
        return scored_segments
    
    def get_top_segments(
        self,
        scored_segments: List[SegmentFeatures],
        top_n: int = 5,
        min_score: float = 0.3
    ) -> List[SegmentFeatures]:
        """
        Get top N segments by fused score.
        
        Args:
            scored_segments: List of scored segments
            top_n: Number of top segments to return
            min_score: Minimum fused score threshold
            
        Returns:
            Top segments sorted by fused score
        """
        # Filter by minimum score
        filtered = [s for s in scored_segments if s.fused_score >= min_score]
        
        # Sort by fused score descending
        sorted_segments = sorted(filtered, key=lambda x: x.fused_score, reverse=True)
        
        return sorted_segments[:top_n]
    
    def explain_score(self, segment: SegmentFeatures) -> str:
        """
        Generate human-readable explanation of a segment's score.
        
        Args:
            segment: Scored segment
            
        Returns:
            Explanation string
        """
        explanation = f"**Fused Score: {segment.fused_score:.2f}**\n"
        explanation += f"- Semantic (what): {segment.semantic_score:.2f}\n"
        explanation += f"- Tonal (how): {segment.tonal_score:.2f}\n"
        explanation += f"- Role (who): {segment.role_relevance:.2f}\n"
        
        if segment.prosodic_features:
            pf = segment.prosodic_features
            explanation += f"\nProsodic Details:\n"
            explanation += f"- Urgency: {pf.get('urgency_score', 0):.2f}\n"
            explanation += f"- Emphasis: {pf.get('emphasis_score', 0):.2f}\n"
        
        return explanation
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
