"""
Participant Profile Store.

Stores participant profiles (name → role + weights) in data/participants.json.
Weights are initially inferred by LLM (LaMini-Flan-T5), with a keyword-formula
fallback. Profiles are per-person, so feedback learning adapts weights individually.
"""

import json
import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional

from .role_hierarchy import get_fallback_weights, get_role_description

logger = logging.getLogger(__name__)

# ─── LLM inference (optional) ────────────────────────────────────────────────

_LLM_WEIGHTS_PROMPT = """\
You are an expert at organisational analysis.

Given this meeting participant:
  Role: "{role}"
  Is External: {is_external}

Output ONLY a valid JSON object with exactly four keys: semantic, tonal, role, temporal.
All values must be floats between 0.0 and 1.0 that sum to 1.0.
They represent how much each signal matters when surfacing important transcript segments
FOR THIS PERSON'S PERSPECTIVE:
  semantic  = importance of WHAT was said (content and meaning)
  tonal     = importance of HOW it was said (urgency, emphasis, tone of voice)
  role      = importance of WHO said it (speaker authority as a relevance gate)
  temporal  = importance of historical continuity (links to previous meetings)

Output example:
{{"semantic": 0.40, "tonal": 0.12, "role": 0.30, "temporal": 0.18}}

JSON:"""


def _infer_weights_llm(role: str, is_external: bool, llm_pipeline) -> Optional[Dict[str, float]]:
    """
    Ask the LLM pipeline to infer fusion weights for a given role.

    Args:
        role: Free-text role string.
        is_external: Whether participant is external.
        llm_pipeline: A HuggingFace text2text pipeline already loaded.

    Returns:
        Normalised weight dict, or None if inference fails.
    """
    prompt = _LLM_WEIGHTS_PROMPT.format(role=role, is_external=is_external)
    try:
        output = llm_pipeline(prompt, max_length=80, do_sample=False)
        raw_text = output[0]["generated_text"].strip()

        # Extract JSON — model may prepend explanation text
        json_match = re.search(r"\{[^{}]+\}", raw_text, re.DOTALL)
        if not json_match:
            logger.warning(f"LLM weight inference: no JSON found in output: {raw_text!r}")
            return None

        data = json.loads(json_match.group())

        required_keys = {"semantic", "tonal", "role", "temporal"}
        if not required_keys.issubset(data.keys()):
            logger.warning(f"LLM weight inference: missing keys in {data}")
            return None

        weights = {k: float(data[k]) for k in required_keys}

        # Validate: all positive, sum ≈ 1.0
        if any(v < 0 for v in weights.values()):
            logger.warning(f"LLM weight inference: negative value in {weights}")
            return None

        total = sum(weights.values())
        if total <= 0:
            return None

        # Normalise
        return {k: round(v / total, 4) for k, v in weights.items()}

    except Exception as e:
        logger.warning(f"LLM weight inference failed: {e}")
        return None


# ─── ParticipantStore ─────────────────────────────────────────────────────────

class ParticipantStore:
    """
    Local backend for participant profiles.

    Each profile stores:
      • display_name, role, department, is_external
      • weights: {semantic, tonal, role, temporal}  — auto-derived on registration
      • authority_score: float derived from role string
      • added_at / updated_at: ISO timestamps

    Profiles are keyed by a normalised name slug (lowercase, spaces → underscores,
    titles stripped) so fuzzy lookup works reliably.
    """

    def __init__(self, data_dir: str = "data", llm_pipeline=None):
        """
        Args:
            data_dir:     Directory for persistence files.
            llm_pipeline: Optional HuggingFace text2text pipeline for weight inference.
                          If None, falls back to keyword formula immediately.
        """
        self.data_dir = data_dir
        self.llm_pipeline = llm_pipeline
        self._profiles_path = os.path.join(data_dir, "participants.json")
        os.makedirs(data_dir, exist_ok=True)
        self._profiles: Dict[str, dict] = self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    def add_participant(
        self,
        name: str,
        role: str,
        is_external: bool = False,
        department: str = "",
        weights: Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Register a participant. If weights are not supplied, derives them
        via LLM (primary) or keyword formula (fallback).

        Returns the saved profile dict.
        """
        slug = self._slugify(name)

        if weights is None:
            weights = self._derive_weights(role, is_external)

        from .role_hierarchy import get_role_authority
        authority = get_role_authority(role)

        profile = {
            "display_name": name,
            "role": role,
            "department": department,
            "is_external": is_external,
            "authority_score": round(authority, 4),
            "weights": weights,
            "weight_source": "llm" if self.llm_pipeline else "formula",
            "added_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        self._profiles[slug] = profile
        self._save()
        logger.info(
            f"ParticipantStore: registered '{name}' as '{role}' "
            f"(external={is_external}) weights={weights}"
        )
        return profile

    def get_participant(self, name: str) -> Optional[dict]:
        """
        Fuzzy lookup by name. Strips honorifics and compares slugs.

        Returns the profile dict or None if not found.
        """
        slug = self._slugify(name)

        # Exact slug match
        if slug in self._profiles:
            return self._profiles[slug]

        # Partial slug match (e.g. "alice" matches "alice_johnson")
        for stored_slug, profile in self._profiles.items():
            if slug in stored_slug or stored_slug in slug:
                return profile

        return None

    def get_weights_for_speaker(self, name: str) -> Dict[str, float]:
        """
        Main entrypoint for FusionLayer.

        Returns profile weights if a match is found, otherwise derives
        weights from keyword fallback (neutral 0.5 authority).

        Args:
            name: Speaker name or diarization label.

        Returns:
            Weight dict {semantic, tonal, role, temporal}.
        """
        profile = self.get_participant(name)
        if profile:
            logger.debug(f"ParticipantStore: profile hit for '{name}'")
            return profile["weights"]

        # Unregistered speaker — use neutral fallback
        logger.debug(f"ParticipantStore: no profile for '{name}', using fallback")
        return get_fallback_weights(role="unknown", is_external=False)

    def update_weights(self, name: str, new_weights: Dict[str, float]) -> bool:
        """
        Persist updated weights (e.g. from online feedback loop) to a profile.

        Args:
            name: Speaker name.
            new_weights: Updated weight dict.

        Returns:
            True if profile was found and updated, False otherwise.
        """
        slug = self._slugify(name)
        profile = self.get_participant(name)
        if not profile:
            logger.warning(f"ParticipantStore.update_weights: no profile for '{name}'")
            return False

        # Find the stored slug (may differ from query slug via partial match)
        stored_slug = next(
            (s for s in self._profiles if s == slug or slug in s or s in slug), None
        )
        if stored_slug is None:
            return False

        self._profiles[stored_slug]["weights"] = new_weights
        self._profiles[stored_slug]["updated_at"] = datetime.utcnow().isoformat()
        self._profiles[stored_slug]["weight_source"] = "feedback"
        self._save()
        logger.info(f"ParticipantStore: updated weights for '{name}': {new_weights}")
        return True

    def register_custom_role(self, role: str, authority: float) -> None:
        """
        Dynamically extend the keyword map for an unknown role title.
        Saved to data/custom_roles.json so it persists across sessions.

        Note: affects future add_participant calls (fallback formula path).
        """
        custom_path = os.path.join(self.data_dir, "custom_roles.json")
        existing: dict = {}
        if os.path.exists(custom_path):
            with open(custom_path) as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    pass
        existing[role.lower()] = round(float(authority), 4)
        with open(custom_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info(f"Registered custom role '{role}' with authority {authority}")

    def list_participants(self) -> List[dict]:
        """Return all registered participant profiles."""
        return list(self._profiles.values())

    def get_ui_badge(self, name: str) -> str:
        """
        Returns a short status string for Streamlit display.

        "🟢 Profile matched" or "🟡 Heuristic used"
        """
        if self.get_participant(name):
            return "🟢 Profile matched"
        return "🟡 Heuristic used"

    # ── Internal ───────────────────────────────────────────────────────────

    def _derive_weights(self, role: str, is_external: bool) -> Dict[str, float]:
        """LLM → formula fallback weight derivation."""
        if self.llm_pipeline is not None:
            weights = _infer_weights_llm(role, is_external, self.llm_pipeline)
            if weights:
                return weights
            logger.warning("LLM inference failed — using formula fallback.")
        return get_fallback_weights(role, is_external)

    @staticmethod
    def _slugify(name: str) -> str:
        """
        Normalise a name for storage / lookup.
        Strips common honorifics, lowercases, replaces spaces/punctuation with _.
        """
        # Strip honorifics
        honorifics = r"^(mr|mrs|ms|dr|prof|sir|madam|mx)[\.\s]+"
        cleaned = re.sub(honorifics, "", name.strip(), flags=re.IGNORECASE)
        cleaned = cleaned.lower()
        cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
        return cleaned.strip("_")

    def _load(self) -> Dict[str, dict]:
        if os.path.exists(self._profiles_path):
            with open(self._profiles_path) as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.warning("participants.json corrupted — starting fresh.")
        return {}

    def _save(self) -> None:
        with open(self._profiles_path, "w") as f:
            json.dump(self._profiles, f, indent=2)
