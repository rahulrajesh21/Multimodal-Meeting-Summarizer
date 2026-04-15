import logging
from typing import Optional, Dict
import requests
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMSummarizer:
    """
    Generates coherent, role-specific summaries and extracts structured JSON
    using a local LM Studio instance (OpenAI-compatible API).
    """

    def __init__(self, model_name: str = "qwen3-4b", endpoint: str = "http://localhost:1234/v1/chat/completions"):
        self.model_name = model_name
        self.endpoint = endpoint
        self.is_ready = False
        self.last_error = None
        self._check_connection()

    def _check_connection(self):
        """Check if the LM Studio endpoint is running."""
        self.last_error = None
        try:
            # Hit the /v1/models endpoint to verify LM Studio is running
            base_url = self.endpoint.rsplit('/v1/', 1)[0]
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                self.is_ready = True
                logger.info(f"LLM Summarizer ready (LM Studio: {self.model_name})")
            else:
                self.is_ready = False
                self.last_error = "LM Studio endpoint returned non-200 status"
        except Exception as e:
            self.last_error = f"Connection to LM Studio failed: {str(e)[:100]}"
            self.is_ready = False
            logger.warning(f"LM Studio connection check failed: {e}. Ensure LM Studio is running.")

    def status(self) -> str:
        """Human-readable load status for UI display."""
        self._check_connection()
        if self.is_ready:
            return f"✅ Ready (LM Studio: {self.model_name})"
        if self.last_error:
            return f"❌ Failed: {self.last_error}"
        return "⏳ Not initialised"

    def _call_ollama(self, prompt, format_json: bool = False, temperature: float = 0.3) -> str:
        """Make a request to LM Studio's OpenAI-compatible chat completions endpoint."""
        
        # Determine the model name. Try to get the first running model, fallback to local-model
        model_name = "local-model"
        try:
            base_url = self.endpoint.rsplit('/v1/', 1)[0]
            models_resp = requests.get(f"{base_url}/v1/models", timeout=2)
            if models_resp.status_code == 200 and models_resp.json().get("data"):
                model_name = models_resp.json()["data"][0]["id"]
        except Exception:
            pass

        payload = {
            "model": model_name,
            "temperature": temperature,
            "stream": False,
        }

        if isinstance(prompt, list):
            payload["messages"] = prompt
        else:
            payload["messages"] = [
                {"role": "user", "content": prompt}
            ]

        if format_json:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            raise e

    def extract_json_events(self, text: str, speaker: Optional[str] = None, timestamp: Optional[str] = None) -> Dict:
        """
        Layer 1: Universal Event + Entity Extraction.
        Extracts structured events and entities from a transcript chunk.
        """
        schema = '''
{
  "events": [
    {
      "event_type": "decision | problem | update | idea | deadline | metric | discussion | risk",
      "summary": "...",
      "entities": [
        { "type": "component | person | feature | metric | topic | client | date", "value": "..." }
      ],
      "sentiment": 0.0, // float from -1.0 to 1.0. Negative for problems, positive for solutions.
      "confidence": 0.0, // float from 0.0 to 1.0
      "timestamp": "...", // use provided timestamp or empty
      "speaker": "..." // use provided speaker or empty
    }
  ]
}
'''
        prompt = f"""
You are an advanced information extraction system processing meeting transcripts.
Analyze the following transcript chunk and extract the core events, topics, and decisions.

Follow these strict rules:
1. Output MUST be valid JSON matching the exact schema provided.
2. If no meaningful events occurred, output an empty events array: {{"events": []}}
3. Do not include introductory text or markdown formatting outside the JSON object.

Speaker: {speaker or 'Unknown'}
Timestamp: {timestamp or 'Unknown'}

Transcript Chunk:
{text}

Expected JSON Schema:
{schema}
"""
        response_text = ""
        try:
            response_text = self._call_ollama(prompt, format_json=True, temperature=0.1)
            
            # Clean up potential markdown formatting that Ollama might still output
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
                
            data = json.loads(cleaned_text.strip())
            
            # Force inject speaker and timestamp if missing from the LLM output
            for event in data.get("events", []):
                if not event.get("speaker") or event["speaker"] == "...":
                    event["speaker"] = speaker or "Unknown"
                if not event.get("timestamp") or event["timestamp"] == "...":
                    event["timestamp"] = timestamp or "Unknown"
                    
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM: {e}\\nRaw output: {response_text}")
            return {"events": []}
        except Exception as e:
            logger.error(f"Event extraction failed: {e}")
            return {"events": []}

    def _build_temporal_prefix(self, temporal_context: dict) -> str:
        """Build a short cross-meeting alert block from temporal_context."""
        lines = []

        # Handling threads (from previous ThreadDetector approach if still used)
        threads = temporal_context.get("threads", [])
        for t in threads[:3]:
            annotation = t.get("annotation") or (
                f"⚠️ Revisited topic — first raised in '{t.get('first_meeting', 'previous meeting')}'"
            )
            lines.append(annotation)

        # Handling related entities from new TemporalGraphMemory
        if not lines:
            entities = temporal_context.get("cross_meeting_entities", [])
            for entity in entities[:3]:
                lines.append(
                    f"⚠️ Recurring {entity.get('type', 'topic')}: {entity.get('name', '')} (Mentioned {entity.get('mentions', 0)} times before)"
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
        """
        if not text:
            return "[Empty text provided for summary]"

        # Strip timestamps from segments so it reads clean prose
        clean_lines = []
        for line in text.strip().splitlines():
            clean = re.sub(r"^\[\d+\.?\d*s\]\s*", "", line.strip())
            if clean:
                clean_lines.append(clean)
        clean_text = " ".join(clean_lines)

        temporal_prefix = self._build_temporal_prefix(temporal_context or {})

        prompt = f"""
You are an expert AI assistant tasked with summarizing meeting transcripts.
Your target audience is a person in the following role: {role}
Your primary focus should be on: {focus}

Here is context from previous meetings that you should incorporate if relevant:
{temporal_prefix}

Please summarize the following transcript chunk. Make sure the summary is concise, professional, and directly addresses the perspective of a {role}.
Do not include any introductory filler text like "Here is the summary". Just provide the summary directly.

Transcript:
{clean_text}
"""
        try:
            summary = self._call_ollama(prompt, temperature=0.3)
            
            # Prepend a small role header + any temporal alerts
            header = f"[{role}] "
            return header + temporal_prefix + summary.strip()

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            reason = str(e)
            return temporal_prefix + f"[LLM unavailable — {reason}]"
