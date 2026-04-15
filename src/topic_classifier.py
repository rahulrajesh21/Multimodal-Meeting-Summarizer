"""
Topic Classifier — Fine-Tuned DistilBERT Inference Module

Loads the distilled DistilBERT model from models/topic_classifier/ and provides
fast, deterministic topic classification for meeting transcript segments.

Classes: decision, discussion, idea, problem, risk, update
"""

import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve model path relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJECT_ROOT / "models" / "topic_classifier"

VALID_TYPES = {"decision", "discussion", "idea", "problem", "risk", "update"}


class TopicClassifier:
    """
    Wraps a fine-tuned DistilBERT model for meeting topic classification.
    Falls back to 'discussion' if the model is not available.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(_MODEL_DIR)
        self.pipeline = None
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        """Load the HuggingFace text-classification pipeline."""
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Topic classifier model not found at {self.model_path}. "
                "Falling back to 'discussion' for all classifications."
            )
            return

        try:
            from transformers import pipeline as hf_pipeline
            self.pipeline = hf_pipeline(
                "text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                top_k=None,  # return all scores
                device="mps",  # Apple Silicon GPU
            )
            self.is_ready = True
            logger.info(f"TopicClassifier loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load topic classifier: {e}. Trying CPU fallback...")
            try:
                from transformers import pipeline as hf_pipeline
                self.pipeline = hf_pipeline(
                    "text-classification",
                    model=self.model_path,
                    tokenizer=self.model_path,
                    top_k=None,
                    device="cpu",
                )
                self.is_ready = True
                logger.info(f"TopicClassifier loaded on CPU from {self.model_path}")
            except Exception as e2:
                logger.error(f"TopicClassifier failed to load on CPU as well: {e2}")

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify a single transcript line.
        Returns (label, confidence) tuple.
        """
        if not self.is_ready or not self.pipeline:
            return ("discussion", 0.0)

        try:
            results = self.pipeline(text[:512])  # DistilBERT max 512 tokens
            if results and isinstance(results[0], list):
                # top_k=None returns list of lists
                top = max(results[0], key=lambda x: x["score"])
            elif results:
                top = max(results, key=lambda x: x["score"])
            else:
                return ("discussion", 0.0)

            label = top["label"].lower()
            if label not in VALID_TYPES:
                label = "discussion"
            return (label, top["score"])
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ("discussion", 0.0)

    def classify_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Classify a batch of transcript lines.
        Returns list of (label, confidence) tuples.
        """
        if not self.is_ready or not self.pipeline or not texts:
            return [("discussion", 0.0)] * len(texts)

        try:
            # Truncate each text to 512 chars for safety
            truncated = [t[:512] for t in texts]
            all_results = self.pipeline(truncated, batch_size=32)

            output = []
            for results in all_results:
                if isinstance(results, list):
                    top = max(results, key=lambda x: x["score"])
                else:
                    top = results
                label = top["label"].lower()
                if label not in VALID_TYPES:
                    label = "discussion"
                output.append((label, top["score"]))
            return output
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            return [("discussion", 0.0)] * len(texts)
