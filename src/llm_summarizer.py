import logging
from typing import Optional
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMSummarizer:
    """
    Generates coherent, role-specific summaries using facebook/bart-large-cnn.
    Loads the model directly (bypasses the pipeline() task-name API which is
    broken for seq2seq in newer transformers versions).
    """

    def __init__(self, device: str = "cpu", model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None   # kept for compatibility checks elsewhere
        self.is_ready = False
        self.last_error = None

        # Resolve torch device
        if device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        self._initialize_model()

    def _initialize_model(self):
        self.last_error = None
        try:
            from transformers import BartForConditionalGeneration, BartTokenizer
            logger.info(f"Loading {self.model_name} tokenizer...")
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loading {self.model_name} model...")
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self._device)
            self.model.eval()
            self.is_ready = True
            logger.info(f"LLM Summarizer ready on {self._device}.")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error initializing LLM Summarizer: {e}")
            self.is_ready = False

    def status(self) -> str:
        """Human-readable load status for UI display."""
        if self.is_ready:
            return f"✅ Ready ({self.model_name} on {self._device})"
        if self.last_error:
            return f"❌ Failed: {self.last_error[:150]}"
        return "⏳ Not initialised"

    def _build_temporal_prefix(self, temporal_context: dict) -> str:
        """
        Build a short cross-meeting alert block from temporal_context.

        Args:
            temporal_context: dict produced by ThreadDetector.get_threads_for_summary()
                              OR TemporalGraphMemory.get_context_for_segment().
                              Expects keys: 'threads' (list) OR 'related_topics' (list).

        Returns:
            Multi-line string with ⚠️ alerts (empty string if no relevant context).
        """
        lines = []

        # Prefer pre-built thread annotations (from ThreadDetector)
        threads = temporal_context.get("threads", [])
        for t in threads[:3]:          # cap at 3 to keep prompts short
            annotation = t.get("annotation") or (
                f"⚠️ Revisited topic — first raised in '{t.get('first_meeting', 'previous meeting')}'"
            )
            lines.append(annotation)

        # Fallback: related_topics list from get_context_for_segment()
        if not lines:
            for topic in temporal_context.get("related_topics", [])[:3]:
                date_str = topic.get("meeting_date", "")[:10]
                lines.append(
                    f"⚠️ Related topic from {date_str}: {topic.get('topic', '')[:80]}"
                )

        if not lines:
            return ""

        prefix = "[Cross-Meeting Context]\n" + "\n".join(lines) + "\n\n"
        return prefix

    def summarize(
        self,
        text: str,
        role: str,
        focus: str = "key decisions and action items",
        temporal_context: Optional[dict] = None,
    ) -> str:
        """
        Generate a role-specific summary from the given text.

        Args:
            text:             Transcript or highlight text to summarise.
            role:             Target role for context (e.g. "CTO").
            focus:            What to pay attention to.
            temporal_context: Optional dict with 'threads' or 'related_topics' keys
                              (from ThreadDetector / TemporalGraphMemory).  When
                              provided, cross-meeting alert lines are prepended to
                              the output so users immediately see recurring issues.

        Returns:
            Summary string (with optional ⚠️ cross-meeting header).
        """
        if not self.is_ready or not text:
            reason = self.last_error or "model not loaded"
            temporal_prefix = self._build_temporal_prefix(temporal_context or {})
            return temporal_prefix + f"[LLM unavailable — {reason}]"

        # Strip timestamps from segments so BART reads clean prose
        # e.g. "[11.0s] So I've got..." → "So I've got..."
        import re
        clean_lines = []
        for line in text.strip().splitlines():
            clean = re.sub(r"^\[\d+\.?\d*s\]\s*", "", line.strip())
            if clean:
                clean_lines.append(clean)
        clean_text = " ".join(clean_lines)

        # BART works purely on the text body — no instruction prefix
        # (it ignores instructions and just echoes them back into the summary)
        input_text = clean_text[:1800]

        try:
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            ).to(self._device)

            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    max_length=220,
                    min_length=60,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Build temporal prefix (cross-meeting alerts)
            temporal_prefix = self._build_temporal_prefix(temporal_context or {})

            # Prepend a small role header + any temporal alerts
            header = f"[{role}] "
            return header + temporal_prefix + summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Error generating summary: {e}"

