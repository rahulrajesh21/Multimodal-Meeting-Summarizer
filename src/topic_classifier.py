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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJECT_ROOT / "models" / "topic_classifier"

VALID_TYPES = {"decision", "discussion", "idea", "problem", "risk", "update"}


def _safe_pipeline_call(pipeline_fn, texts):
    """
    Call a HuggingFace pipeline while suppressing `token_type_ids` errors
    that occur with DistilBERT on newer transformers versions.
    Returns raw pipeline output.
    """
    try:
        return pipeline_fn(texts)
    except TypeError as e:
        if "token_type_ids" not in str(e):
            raise
        # Patch: call model/tokenizer directly without token_type_ids
        import torch
        tokenizer = pipeline_fn.tokenizer
        model = pipeline_fn.model
        is_batch = isinstance(texts, list)
        inputs_list = texts if is_batch else [texts]
        results = []
        for text in inputs_list:
            enc = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
            enc.pop("token_type_ids", None)  # remove the offending key
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            id2label = model.config.id2label
            scored = [{"label": id2label[i], "score": p} for i, p in enumerate(probs)]
            results.append(scored)
        return results if is_batch else results


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
            # Use CPU — MPS has allocation issues with this DistilBERT checkpoint
            self.pipeline = hf_pipeline(
                "text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                top_k=None,
                device="cpu",
            )
            self.is_ready = True
            logger.info(f"TopicClassifier loaded on CPU from {self.model_path}")
        except Exception as e:
            logger.error(f"TopicClassifier failed to load: {e}")



    def classify(self, text: str) -> Tuple[str, float]:
        """Classify a single transcript line. Returns (label, confidence)."""
        if not self.is_ready or not self.pipeline:
            return ("discussion", 0.0)

        try:
            results = _safe_pipeline_call(self.pipeline, text[:512])
            # Normalise: may be list-of-list or list-of-dict
            inner = results[0] if results and isinstance(results[0], list) else results
            top = max(inner, key=lambda x: x["score"])
            label = top["label"].lower()
            if label not in VALID_TYPES:
                label = "discussion"
            return (label, top["score"])
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ("discussion", 0.0)


    def classify_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Classify a batch of transcript lines. Returns list of (label, confidence)."""
        if not self.is_ready or not self.pipeline or not texts:
            return [("discussion", 0.0)] * len(texts)

        try:
            truncated = [t[:512] for t in texts]
            all_results = _safe_pipeline_call(self.pipeline, truncated)

            output = []
            for results in all_results:
                inner = results if isinstance(results, list) else [results]
                top = max(inner, key=lambda x: x["score"])
                label = top["label"].lower()
                if label not in VALID_TYPES:
                    label = "discussion"
                output.append((label, top["score"]))
            return output
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            return [("discussion", 0.0)] * len(texts)

