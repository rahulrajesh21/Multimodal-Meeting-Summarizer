import logging
from typing import List, Optional, Dict
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

    def _get_model_name(self, preferred_substrings: list) -> str:
        """Dynamically pulls the best model from LM Studio's loaded models."""
        model_name = "local-model"
        try:
            base_url = self.endpoint.rsplit('/v1/', 1)[0]
            models_resp = requests.get(f"{base_url}/v1/models", timeout=2)
            if models_resp.status_code == 200 and models_resp.json().get("data"):
                models = models_resp.json()["data"]
                # 1. Look for a preferred model explicitly
                for pref in preferred_substrings:
                    for m in models:
                        if pref.lower() in m["id"].lower():
                            return m["id"]
                # 2. Fallback to the first available
                return models[0]["id"]
        except Exception:
            pass
        return model_name

    def _call_ollama(self, prompt, format_json: bool = False, temperature: float = 0.3) -> str:
        """Make a request to LM Studio's OpenAI-compatible chat completions endpoint."""
        
        # Target light, fast background models for extraction
        model_name = self._get_model_name(["ministral", "qwen2.5-3b"])

        payload = {
            "model": model_name,
            "temperature": temperature,
            "stream": False,
            "reasoning_effort": "none"
        }

        if isinstance(prompt, list):
            payload["messages"] = prompt
        else:
            payload["messages"] = [
                {"role": "user", "content": prompt}
            ]

        # NOTE: response_format is not supported by all models (e.g. Gemma).
        # The prompt already instructs JSON output, so this is optional.
        # if format_json:
        #     payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            raise e

    def agent_chat(self, messages: List[Dict], tools: List[Dict], tool_handler: callable, max_turns: int = 5, step_callback: callable = None) -> Dict:
        """
        Agent orchestration loop with real-time streaming.
        Uses stream=True to get token-by-token reasoning and content.
        Returns a dict: { "reply": str, "steps": [...], "model": str }
        If step_callback is provided, each step is emitted in real-time.
        """
        import json
        import re as _re
        
        model_name = self._get_model_name(["gemma", "deepseek"])
        steps: List[Dict] = []

        def _emit(step: Dict):
            if step.get("type") == "thinking" and step.get("status") == "done":
                for i in range(len(steps)-1, -1, -1):
                    if steps[i].get("type") == "thinking" and steps[i].get("turn") == step.get("turn"):
                        steps[i] = step
                        break
                else:
                    steps.append(step)
            else:
                steps.append(step)
            
            if step_callback:
                step_callback(step)

        for turn in range(max_turns):
            payload = {
                "model": model_name,
                "temperature": 0.4,
                "stream": True,
                "messages": messages,
                "tools": tools,
            }

            try:
                # ── Stream the LLM response ──────────────────────────
                resp = requests.post(self.endpoint, json=payload, timeout=120, stream=True)
                resp.raise_for_status()

                # Accumulators
                reasoning_buf = ""
                content_buf = ""
                tool_calls_buf: Dict[int, Dict] = {}  # index → {id, name, arguments}
                reasoning_emitted = False

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data: "):
                        continue
                    data_str = raw_line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})

                    # -- Reasoning / thinking tokens --
                    reasoning_content = delta.get("reasoning_content", "")
                    if reasoning_content:
                        if not reasoning_emitted:
                            # Emit a "thinking" step immediately so the UI shows it
                            _emit({"type": "thinking", "content": "", "turn": turn, "status": "started"})
                            reasoning_emitted = True
                        reasoning_buf += reasoning_content

                    # -- Content tokens (final response or <think> tags) --
                    content_piece = delta.get("content", "")
                    if content_piece:
                        content_buf += content_piece

                    # -- Tool call chunks (streamed incrementally) --
                    tc_deltas = delta.get("tool_calls", [])
                    for tc in tc_deltas:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_buf:
                            tool_calls_buf[idx] = {
                                "id": tc.get("id", ""),
                                "name": "",
                                "arguments": "",
                            }
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_calls_buf[idx]["name"] += fn["name"]
                        if fn.get("arguments"):
                            tool_calls_buf[idx]["arguments"] += fn["arguments"]
                        if tc.get("id"):
                            tool_calls_buf[idx]["id"] = tc["id"]

                resp.close()

                # ── Post-process accumulated content ─────────────────
                # Handle <think> tags in content (some models embed thinking here)
                think_match = _re.search(r'<think>(.*?)</think>', content_buf, _re.DOTALL)
                if think_match:
                    reasoning_buf = think_match.group(1).strip()
                    content_buf = _re.sub(r'<think>.*?</think>', '', content_buf, flags=_re.DOTALL).strip()

                # Emit final thinking content (update with full text)
                if reasoning_buf:
                    if not reasoning_emitted:
                        _emit({"type": "thinking", "content": reasoning_buf, "turn": turn})
                    else:
                        # Update the thinking step with full content
                        _emit({"type": "thinking", "content": reasoning_buf, "turn": turn, "status": "done"})

                # ── Build assistant message for conversation history ──
                message_out: Dict = {"role": "assistant", "content": content_buf}
                if tool_calls_buf:
                    message_out["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in tool_calls_buf.values()
                    ]

                messages.append(message_out)

                # ── Handle tool calls ────────────────────────────────
                if tool_calls_buf:
                    for idx, tc in tool_calls_buf.items():
                        fn_name = tc["name"]
                        
                        allowed_tools = [t["function"]["name"] for t in tools]
                        if fn_name not in allowed_tools:
                            result_text = f"Error: Tool '{fn_name}' is not recognized."
                            _emit({"type": "tool_call", "name": fn_name, "args": {}, "error": result_text, "turn": turn})
                            messages.append({
                                "role": "tool", "tool_call_id": tc["id"],
                                "name": fn_name, "content": str(result_text)
                            })
                            continue

                        try:
                            args = json.loads(tc["arguments"])
                        except json.JSONDecodeError as e:
                            result_text = f"Error: Invalid JSON in arguments: {e}"
                            _emit({"type": "tool_call", "name": fn_name, "args_raw": tc["arguments"], "error": result_text, "turn": turn})
                            messages.append({
                                "role": "tool", "tool_call_id": tc["id"],
                                "name": fn_name, "content": str(result_text)
                            })
                            continue

                        logger.info(f"LLM Agent calling tool: {fn_name} with args: {args}")
                        _emit({"type": "tool_call", "name": fn_name, "args": args, "turn": turn})

                        try:
                            result_text = tool_handler(fn_name, args)
                        except Exception as e:
                            logger.error(f"Tool {fn_name} failed: {e}")
                            result_text = f"Tool {fn_name} error: {str(e)}"

                        _emit({"type": "tool_result", "name": fn_name, "result": str(result_text)[:500], "turn": turn})

                        messages.append({
                            "role": "tool", "tool_call_id": tc["id"],
                            "name": fn_name, "content": str(result_text)
                        })

                    continue  # Loop back for the LLM to process tool results
                else:
                    return {"reply": content_buf, "steps": steps, "model": model_name}

            except Exception as e:
                logger.error(f"Agent chat failed at turn {turn}: {e}")
                raise e

        return {"reply": "Agent stopped: Reached maximum tool turns.", "steps": steps, "model": model_name}

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
4. CRITICAL: Ignore ALL backchanneling, acknowledgments, and conversational filler
   (e.g., "okay", "yeah", "mm-hmm", "right", "sure", "sounds good", "uh-huh").
   These are NOT topics, entities, or events. Return {{"events": []}} for them.
5. Do NOT create entities named "acknowledgment", "agreement", "response",
   "affirmation", "meeting flow", or any abstract conversational label.
   Only extract concrete, product-related or technically substantive entities.
6. Do not add comments (// ...) inside the JSON output.
7. If the text contains [Previous], [Current], [Next] labels, focus extraction
   on the [Current] segment but use surrounding context for pronoun resolution.
   Resolve "it", "that", "the issue" to their concrete referent when possible.

--- FEW-SHOT EXAMPLES ---
Example 1 (filler only):
Input: "Mm-hmm. Yeah. Okay."
Output: {{"events": []}}

Example 2 (substantive):
Input: "We should use an LCD screen for the display. That'll cost about twelve fifty per unit."
Output: {{"events": [{{"event_type": "decision", "summary": "Decided to use LCD screen for display at 12.50 per unit", "entities": [{{"type": "component", "value": "LCD screen"}}, {{"type": "metric", "value": "unit cost 12.50"}}], "sentiment": 0.3, "confidence": 0.9, "timestamp": "", "speaker": ""}}]}}

Example 3 (context window with pronoun):
Input: "[Previous] We discussed the remote control buttons.\\n[Current] Yeah I think it should be larger.\\n[Next] And maybe backlit."
Output: {{"events": [{{"event_type": "idea", "summary": "Suggested making remote control buttons larger", "entities": [{{"type": "feature", "value": "button size"}}, {{"type": "component", "value": "remote control"}}], "sentiment": 0.2, "confidence": 0.7, "timestamp": "", "speaker": ""}}]}}
--- END EXAMPLES ---

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
            
            # Clean up: thinking models may output reasoning before JSON.
            # Extract the first valid JSON object from the response.
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            # Find the first '{' and last '}' to extract JSON from thinking output
            first_brace = cleaned_text.find("{")
            last_brace = cleaned_text.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                cleaned_text = cleaned_text[first_brace:last_brace + 1]
            
            # Strip // comments that LLMs sometimes add (invalid JSON)
            cleaned_text = re.sub(r'//[^\n]*', '', cleaned_text)
                
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

        # New format: pre-formatted LLM context with citations
        llm_context = temporal_context.get("llm_context", "")
        if llm_context:
            return llm_context + "\n"

        # Handling cross-meeting entities with provenance
        entities = temporal_context.get("cross_meeting_entities", [])
        for entity in entities[:5]:
            citation = entity.get('citation', '')
            state = ""
            if entity.get('unresolved_score', 0) > 0.6:
                state = " ⚠️ UNRESOLVED"
            elif entity.get('event_type') == 'decision':
                state = " ✅ DECIDED"

            lines.append(
                f"- [{entity.get('event_type', 'topic').upper()}] "
                f"{entity.get('entity', entity.get('name', ''))}: "
                f"{entity.get('summary', entity.get('text', ''))[:120]}"
                f"{state} {citation}"
            )

        # Fallback: old thread format
        if not lines:
            threads = temporal_context.get("threads", [])
            for t in threads[:3]:
                annotation = t.get("annotation") or (
                    f"⚠️ Revisited topic — first raised in '{t.get('first_meeting', 'previous meeting')}'"
                )
                lines.append(annotation)

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
        scored_segments: Optional[list] = None,
        tm=None,
    ) -> str:
        """
        Generate a role-specific summary from the given text.
        Injects historical context when temporal memory is provided.
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

        # ── Graph Enrichment (The "Batch" Injection) ─────────────
        temporal_prefix = ""
        try:
            if tm:
                # Extract top 3-5 entities from the clean prose
                entities = tm._extract_topic_labels(clean_text)
                
                # Deduplicate and query top entities
                contexts = []
                seen_entities = set()
                for ent in entities:
                    ent_lower = ent.lower().strip()
                    if ent_lower not in seen_entities:
                        seen_entities.add(ent_lower)
                        ctx = tm.format_context_for_llm(ent, top_k=2)
                        if ctx:
                            contexts.append(ctx.replace("[Cross-Meeting Context]", "").strip())
                
                if contexts:
                    full_context = "[Cross-Meeting Context]\n" + "\n\n".join(contexts) + "\n\n"
                    # Truncate if necessary (~800 tokens max)
                    if len(full_context) > 2500:
                        full_context = full_context[:2500] + "\n...[truncated]" + "\n\n"
                    temporal_prefix = full_context
        except Exception as e:
            logger.warning(f"Failed to fetch cross-meeting context for summary: {e}")

        # Fallback to old format
        if not temporal_prefix:
            temporal_prefix = self._build_temporal_prefix(temporal_context or {})

        prompt = f"""
You are an expert AI assistant tasked with summarizing meeting transcripts.
Your target audience is a person in the following role: {role}
Your primary focus should be on: {focus}

{temporal_prefix}

Please summarize the following transcript chunk. Make sure the summary is concise, professional, and directly addresses the perspective of a {role}.
Do not include any introductory filler text like "Here is the summary". Just provide the summary directly.

CRITICAL INSTRUCTION:
Use the provided Cross-Meeting Context (if any) to explain how today's discussion connects to past decisions or unresolved issues.
When referencing cross-meeting history, you MUST use the provided Markdown-style citations (e.g. (Meeting D, Speaker A, 11:32)). Do not hallucinate history or citations.

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
