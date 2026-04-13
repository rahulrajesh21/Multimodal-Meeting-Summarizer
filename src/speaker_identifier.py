"""
SpeakerIdentifier — Automatic SPEAKER_XX → Participant Name Mapping

Fuses three complementary signals to resolve diarization labels to real names:
  1. Voice Fingerprinting  — pyannote SpeakerEmbedding cosine similarity
  2. Transcript Name Scan  — detects name mentions near each speaker's turns
  3. Order-of-Speech       — positional fallback (SPEAKER_00 → first registered, etc.)

Usage:
    si = SpeakerIdentifier(hf_token="...", device="cpu")
    si.load_voice_prints("data/voice_prints.npz")                 # optional
    mapping = si.build_mapping(
        diarization_segments,   # [{start, end, speaker, text?}]
        audio_array,            # float32 waveform
        sample_rate,
        registered_participants,  # ordered list of names from ParticipantStore
        transcript_text,
    )
    # mapping → {"SPEAKER_00": ("Lily", 0.92), "SPEAKER_01": ("Craig", 0.55), ...}
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pyannote.audio import Inference, Model
    PYANNOTE_EMBED_AVAILABLE = True
except ImportError:
    PYANNOTE_EMBED_AVAILABLE = False
    logger.info("pyannote.audio Inference not available — voice fingerprinting disabled.")


# ── Constants ──────────────────────────────────────────────────────────────────
VOICE_SIMILARITY_THRESHOLD = 0.75   # cosine similarity to claim a voice match
LOW_CONFIDENCE_THRESHOLD   = 0.60   # below this → warn user
ORDER_BASE_CONFIDENCE      = 0.45   # heuristic positional mapping confidence
TRANSCRIPT_CONFIDENCE_CAP  = 0.85   # max confidence from name-scan signal
FUSION_WEIGHTS = {
    "voice":      0.55,
    "transcript": 0.30,
    "order":      0.15,
}


class SpeakerIdentifier:
    """
    Automatic SPEAKER_XX → participant name resolver.

    Args:
        hf_token:  Hugging Face token (needed only for voice fingerprinting model download).
        device:    'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "cpu",
    ):
        self.hf_token = hf_token
        self.device   = device

        # voice_prints: {name: embedding_np}
        self._voice_prints: Dict[str, np.ndarray] = {}

        # pyannote embedding model (lazy-loaded)
        self._embed_model = None
        self._embed_infer = None
        self._embed_failed = False   # avoid repeated failed loads

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_voice_print(
        self,
        name: str,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
    ) -> bool:
        """
        Compute and store a voice embedding for a participant.

        Args:
            name:         Participant's display name.
            audio_array:  Float32 mono audio (at least 1 second recommended).
            sample_rate:  Audio sample rate.

        Returns:
            True if embedding was computed successfully, False otherwise.
        """
        emb = self._embed_audio(audio_array, sample_rate)
        if emb is None:
            logger.warning(f"register_voice_print: could not embed audio for '{name}'")
            return False
        self._voice_prints[name] = emb
        logger.info(f"Voice print registered for '{name}' (dim={emb.shape})")
        return True

    def load_voice_prints(self, path: str) -> None:
        """Load persisted voice prints from a .npz file."""
        import os
        if not os.path.exists(path):
            return
        try:
            data = np.load(path, allow_pickle=False)
            self._voice_prints = {name: data[name] for name in data.files}
            logger.info(f"Loaded {len(self._voice_prints)} voice prints from {path}")
        except Exception as e:
            logger.warning(f"Failed to load voice prints from {path}: {e}")

    def save_voice_prints(self, path: str) -> None:
        """Persist voice prints to a .npz file."""
        if not self._voice_prints:
            return
        try:
            np.savez(path, **self._voice_prints)
            logger.info(f"Saved {len(self._voice_prints)} voice prints to {path}")
        except Exception as e:
            logger.warning(f"Failed to save voice prints to {path}: {e}")

    def build_mapping(
        self,
        diarization_segments: List[Dict],
        audio_array: np.ndarray,
        sample_rate: int,
        registered_participants: List[str],
        transcript_text: str = "",
    ) -> Dict[str, Tuple[str, float]]:
        """
        Main entrypoint. Resolves SPEAKER_XX labels to participant names.

        Args:
            diarization_segments: List of [{start, end, speaker, text?}] dicts.
            audio_array:          Full meeting audio as float32 numpy array.
            sample_rate:          Audio sample rate (Hz).
            registered_participants: Ordered list of participant names from ParticipantStore.
            transcript_text:      Full transcript string (used for name scanning).

        Returns:
            Dict mapping SPEAKER_XX → (resolved_name, confidence_score 0–1).
            Unknown speakers (no match found) map to (None, 0.0).
        """
        if not diarization_segments:
            return {}

        # Extract unique speaker IDs sorted by first appearance / numeric suffix
        speaker_ids = self._sorted_speaker_ids(diarization_segments)

        if not speaker_ids:
            return {}

        logger.info(f"SpeakerIdentifier: resolving {len(speaker_ids)} speakers: {speaker_ids}")

        # ── Signal 1: Order heuristic (always available) ──────────────────────
        order_map = self._order_heuristic_mapping(speaker_ids, registered_participants)

        # ── Signal 2: Transcript name scan (always available) ─────────────────
        # Build per-speaker text: group transcript lines by speaker
        speaker_text_map = self._build_speaker_text_map(diarization_segments, transcript_text)
        transcript_map = self._transcript_name_mapping(speaker_text_map, registered_participants)

        # ── Signal 3: Voice fingerprinting (optional — needs pyannote + voice prints) ──
        voice_map: Dict[str, Tuple[str, float]] = {}
        if self._voice_prints and PYANNOTE_EMBED_AVAILABLE and TORCH_AVAILABLE:
            speaker_embeddings = self._embed_all_speakers(
                speaker_ids, diarization_segments, audio_array, sample_rate
            )
            if speaker_embeddings:
                voice_map = self._voice_fingerprint_mapping(speaker_embeddings)

        # ── Fuse signals ──────────────────────────────────────────────────────
        final_mapping = self._fuse_signals(speaker_ids, order_map, transcript_map, voice_map)

        for spk, (name, conf) in final_mapping.items():
            badge = "✅" if conf >= LOW_CONFIDENCE_THRESHOLD else "⚠️" if name else "❓"
            logger.info(f"  {badge} {spk} → {name!r}  conf={conf:.2f}")

        return final_mapping

    # ── Signal implementations ─────────────────────────────────────────────────

    def _order_heuristic_mapping(
        self,
        speaker_ids: List[str],
        registered_names: List[str],
    ) -> Dict[str, Tuple[str, float]]:
        """
        Map SPEAKERs to names by position (SPEAKER_00 → name[0], etc.).
        Confidence decays with position to reflect uncertainty.
        """
        mapping: Dict[str, Tuple[str, float]] = {}
        for i, spk in enumerate(speaker_ids):
            if i < len(registered_names):
                # Confidence slightly higher for first few matched speakers
                conf = ORDER_BASE_CONFIDENCE - (i * 0.03)
                conf = max(0.25, conf)
                mapping[spk] = (registered_names[i], conf)
            else:
                mapping[spk] = (None, 0.0)
        return mapping

    def _transcript_name_mapping(
        self,
        speaker_text_map: Dict[str, str],
        registered_names: List[str],
    ) -> Dict[str, Tuple[str, float]]:
        """
        For each speaker, scan their transcript text and nearby context for
        registered participant first names / full names. Count matches and
        normalise to a 0–1 confidence.
        """
        mapping: Dict[str, Tuple[str, float]] = {}

        # Build first-name patterns for quick matching
        name_patterns: Dict[str, re.Pattern] = {}
        for full_name in registered_names:
            first = full_name.strip().split()[0]
            pattern = re.compile(
                rf"\b{re.escape(first)}\b",
                flags=re.IGNORECASE
            )
            name_patterns[full_name] = pattern

        for spk, text in speaker_text_map.items():
            if not text:
                mapping[spk] = (None, 0.0)
                continue

            counts: Dict[str, int] = {}
            for full_name, pat in name_patterns.items():
                counts[full_name] = len(pat.findall(text))

            if not counts or max(counts.values()) == 0:
                mapping[spk] = (None, 0.0)
                continue

            best_name = max(counts, key=counts.get)
            best_count = counts[best_name]

            # Confidence: sigmoid-like based on mention count
            # 1 mention → ~0.55, 3 mentions → ~0.75, 5+ → ~0.85
            conf = min(TRANSCRIPT_CONFIDENCE_CAP, 0.50 + 0.07 * best_count)
            mapping[spk] = (best_name, conf)

        return mapping

    def _voice_fingerprint_mapping(
        self,
        speaker_embeddings: Dict[str, np.ndarray],
    ) -> Dict[str, Tuple[str, float]]:
        """
        Match speaker embeddings against registered voice prints using cosine similarity.
        """
        mapping: Dict[str, Tuple[str, float]] = {}

        for spk, spk_emb in speaker_embeddings.items():
            best_name: Optional[str] = None
            best_sim = -1.0

            for name, vp_emb in self._voice_prints.items():
                sim = self._cosine_similarity(spk_emb, vp_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

            if best_name and best_sim >= VOICE_SIMILARITY_THRESHOLD:
                # Scale confidence: similarity range [threshold, 1.0] → [0.65, 0.99]
                conf = 0.65 + (best_sim - VOICE_SIMILARITY_THRESHOLD) / (1.0 - VOICE_SIMILARITY_THRESHOLD) * 0.34
                mapping[spk] = (best_name, round(min(conf, 0.99), 3))
            else:
                # Below threshold — don't confidently claim a name
                mapping[spk] = (None, 0.0)

        return mapping

    def _fuse_signals(
        self,
        speaker_ids: List[str],
        order_map: Dict[str, Tuple[str, float]],
        transcript_map: Dict[str, Tuple[str, float]],
        voice_map: Dict[str, Tuple[str, float]],
    ) -> Dict[str, Tuple[str, float]]:
        """
        Weighted vote fusion. Each signal casts a weighted vote for a candidate.
        Final answer = candidate with highest weighted confidence sum.
        Handles conflicts (different signals disagree) by trusting weights.
        """
        final: Dict[str, Tuple[str, float]] = {}

        signal_maps = [
            (order_map,      FUSION_WEIGHTS["order"]),
            (transcript_map, FUSION_WEIGHTS["transcript"]),
            (voice_map,      FUSION_WEIGHTS["voice"]),
        ]

        for spk in speaker_ids:
            candidate_scores: Dict[str, float] = defaultdict(float)

            for sig_map, weight in signal_maps:
                if spk not in sig_map:
                    continue
                name, conf = sig_map[spk]
                if name is None or conf <= 0.0:
                    continue
                candidate_scores[name] += conf * weight

            if not candidate_scores:
                final[spk] = (None, 0.0)
                continue

            best = max(candidate_scores, key=candidate_scores.get)
            total_weight = sum(w for _, w in signal_maps)
            fused_conf = round(candidate_scores[best] / total_weight, 3)
            final[spk] = (best, fused_conf)

        return final

    # ── Audio embedding helpers ────────────────────────────────────────────────

    def _get_embed_inference(self):
        """Lazy-load pyannote SpeakerEmbedding model."""
        if self._embed_failed or not PYANNOTE_EMBED_AVAILABLE or not TORCH_AVAILABLE:
            return None
        if self._embed_infer is not None:
            return self._embed_infer
        try:
            model = Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=self.hf_token,
            )
            device = torch.device(self.device if self.device != "mps" else "cpu")
            model = model.to(device)
            self._embed_infer = Inference(model, window="whole")
            logger.info("pyannote SpeakerEmbedding model loaded for voice fingerprinting.")
        except Exception as e:
            logger.warning(f"SpeakerEmbedding model load failed: {e}. Voice fingerprinting disabled.")
            self._embed_failed = True
        return self._embed_infer

    def _embed_audio(self, audio_array: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Get speaker embedding for a raw audio clip."""
        infer = self._get_embed_inference()
        if infer is None:
            return None
        try:
            waveform = torch.from_numpy(audio_array).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            emb = infer({"waveform": waveform, "sample_rate": sample_rate})
            return np.array(emb).flatten()
        except Exception as e:
            logger.warning(f"Audio embedding failed: {e}")
            return None

    def _embed_all_speakers(
        self,
        speaker_ids: List[str],
        segments: List[Dict],
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, np.ndarray]:
        """Compute average embedding per speaker from all their utterances."""
        embeddings: Dict[str, np.ndarray] = {}

        # Group segment intervals by speaker
        spk_intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for seg in segments:
            spk = seg.get("speaker")
            if spk:
                spk_intervals[spk].append((seg["start"], seg["end"]))

        for spk in speaker_ids:
            if spk not in spk_intervals:
                continue
            clip_embs = []
            for start, end in spk_intervals[spk]:
                s = int(start * sample_rate)
                e = int(end   * sample_rate)
                clip = audio_array[s:e]
                if len(clip) < sample_rate * 0.5:   # skip clips < 0.5s
                    continue
                emb = self._embed_audio(clip, sample_rate)
                if emb is not None:
                    clip_embs.append(emb)
            if clip_embs:
                embeddings[spk] = np.mean(clip_embs, axis=0)

        return embeddings

    # ── Utility helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _sorted_speaker_ids(segments: List[Dict]) -> List[str]:
        """
        Return unique speaker IDs sorted by first appearance in segment list,
        with fallback sort by numeric suffix (SPEAKER_00, SPEAKER_01, …).
        """
        seen: List[str] = []
        for seg in segments:
            spk = seg.get("speaker")
            if spk and spk not in seen:
                seen.append(spk)
        return seen

    @staticmethod
    def _build_speaker_text_map(
        segments: List[Dict],
        transcript_text: str,
    ) -> Dict[str, str]:
        """
        Build a dict of {speaker_id: concatenated_text}.
        Uses 'text' field from segments if available, otherwise falls back to
        parsing the full transcript string.
        """
        by_speaker: Dict[str, List[str]] = defaultdict(list)

        # Use segment-level text if available
        for seg in segments:
            spk = seg.get("speaker")
            text = seg.get("text", "").strip()
            if spk and text:
                by_speaker[spk].append(text)

        # Also scan full transcript for SPEAKER_XX lines (format: [HH:MM:SS] [SPEAKER_XX] text)
        if transcript_text:
            line_pattern = re.compile(
                r'\[\d{2}:\d{2}:\d{2}\]\s*\[(SPEAKER_\d{2})\]\s*(.*)'
            )
            for match in line_pattern.finditer(transcript_text):
                spk, text = match.group(1), match.group(2).strip()
                if text and spk not in by_speaker:
                    by_speaker[spk].append(text)

        # Concatenate all text for each speaker into a single searchable string
        return {spk: " ".join(texts) for spk, texts in by_speaker.items()}

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1-D vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
