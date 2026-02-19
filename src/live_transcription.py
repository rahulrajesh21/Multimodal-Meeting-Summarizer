"""
Real-time transcription engine using Faster-Whisper (CTranslate2) or Hugging Face transformers.
Optimized for lightweight usage with Pyannote diarization.
"""

import os
import sys
import threading
import queue
import time
import numpy as np
import torch
from typing import Optional, Iterator

# Import transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

# Import faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not installed. Install with: pip install faster-whisper")

# Try to import pyannote.audio directly for robust diarization
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Install with: pip install pyannote.audio")

# Fix for PyTorch 2.6+ weights_only=True security change
if hasattr(torch, 'load'):
    _original_load = torch.load
    
    def _safe_load(*args, **kwargs):
        # FORCE weights_only=False to support legacy checkpoints (Pyannote)
        if 'weights_only' in kwargs:
            del kwargs['weights_only']
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
        
    torch.load = _safe_load

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_capture import AudioCapture, AudioBuffer


class LiveTranscriber:
    """
    Real-time transcription engine that processes live audio streams.
    Supports 'faster_whisper' (optimized CPU/CUDA) and 'huggingface' (MPS/CUDA/CPU).
    """
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: str = "cpu",
        enable_diarization: bool = False,
        backend: str = "faster_whisper" 
    ):
        """
        Initialize the live transcriber.
        
        Args:
            model_size: Model size for Whisper (tiny, base, small, medium, large-v2, large-v3)
            language: Optional language code (e.g., "en", "es", "fr")
            hf_token: Hugging Face token (required for diarization)
            device: Device to run on ("cpu", "cuda", "mps")
            enable_diarization: Enable speaker diarization
            backend: "faster_whisper" or "huggingface"
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token
        self.backend = backend
        self.diarize_model = None
        
        self.pipe = None
        self.model = None

        print(f"Initializing LiveTranscriber with backend: {backend} on {device}")

        if backend == "faster_whisper":
            if not FASTER_WHISPER_AVAILABLE:
                print("faster-whisper not available, falling back to huggingface")
                self.backend = "huggingface"
            else:
                self._init_faster_whisper()

        if self.backend == "huggingface":
            self._init_huggingface()
            
        # Initialize diarization if enabled
        if enable_diarization:
            self._init_diarization(hf_token, device)
        
        # Transcription state
        self.is_transcribing = False
        self.transcription_thread: Optional[threading.Thread] = None
        self.transcript_queue = queue.Queue()
        self.full_transcript = []

    def _init_faster_whisper(self):
        """Initialize faster-whisper model."""
        # faster-whisper supports "cpu" ("int8") or "cuda" ("float16"). 
        # It does NOT support "mps".
        compute_type = "int8" # Default for CPU
        device_type = "cpu"
        
        if self.device == "cuda":
            device_type = "cuda"
            compute_type = "float16"
        elif self.device == "mps":
             print("Note: faster-whisper does not support MPS directly. Using CPU (optimized with CTranslate2).")
             device_type = "cpu"
             compute_type = "int8"
        
        print(f"Loading faster-whisper model '{self.model_size}' on {device_type} ({compute_type})...")
        try:
            self.model = WhisperModel(
                self.model_size, 
                device=device_type, 
                compute_type=compute_type
            )
            print("faster-whisper model loaded!")
        except Exception as e:
            print(f"Failed to load faster-whisper: {e}")
            raise

    def _init_huggingface(self):
        """Initialize Hugging Face pipeline."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is not installed.")
            
        print(f"Loading Hugging Face Whisper model: openai/whisper-{self.model_size} on {self.device}...")
        
        # Map device string to torch device format for pipeline
        if self.device == "cuda":
            device_arg = "cuda:0"
        elif self.device == "mps":
            device_arg = "mps"
        else:
            device_arg = "cpu"
            
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=f"openai/whisper-{self.model_size}",
                chunk_length_s=30,
                device=device_arg,
            )
            print(f"Hugging Face model loaded successfully on {device_arg}!")
        except Exception as e:
            raise RuntimeError(f"Failed to load Hugging Face pipeline: {e}")

    def _init_diarization(self, hf_token: Optional[str], device: str):
        """Initialize the pyannote diarization pipeline."""
        if not hf_token:
            hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            print("Warning: Diarization requires HF_TOKEN. Set environment variable or pass hf_token parameter.")
            self.enable_diarization = False
            return

        if not PYANNOTE_AVAILABLE:
            print("Warning: pyannote.audio not installed. Diarization disabled.")
            self.enable_diarization = False
            return

        print("Initializing speaker diarization pipeline...")
        
        try:
            # print("Attempting to load pyannote/speaker-diarization-3.1...")
            self.diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
            if self.diarize_model:
                self.diarize_model.to(torch.device(device))
                print("Diarization pipeline loaded successfully!")
            else:
                print("Failed to load pyannote pipeline (returned None). Check HF_TOKEN and access rights.")
                self.enable_diarization = False
        except Exception as e:
            print(f"Error loading pyannote pipeline: {e}")
            self.enable_diarization = False
        
    def transcribe_chunk(self, audio_data, sample_rate: int = 16000) -> str:
        """
        Transcribe a single audio chunk.
        """
        # Convert to float32 array if needed
        audio_array = self._prepare_audio(audio_data)

        try:
            if self.backend == "faster_whisper":
                # faster-whisper transcribe
                # Disable VAD filter for now to avoid onnxruntime issues
                segments, _ = self.model.transcribe(
                    audio_array, 
                    language=self.language,
                    beam_size=5,
                    vad_filter=False 
                )
                # Convert generator to string
                text = " ".join([segment.text for segment in segments]).strip()
                return text

            else: # huggingface
                # Passing return_timestamps=False for short chunks to get just text
                prediction = self.pipe(audio_array, return_timestamps=False)
                
                if isinstance(prediction, dict):
                    return prediction.get("text", "").strip()
                elif isinstance(prediction, list):
                     return " ".join([p.get("text", "") for p in prediction]).strip()
                else:
                     return str(prediction).strip()

        except Exception as e:
            print(f"Transcription failed: {e}")
            return ""

    def _prepare_audio(self, audio_data):
        """Helper to convert audio bytes/numpy to float32 mono."""
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.float32:
                audio_array = audio_data.astype(np.float32)
            else:
                audio_array = audio_data
        else:
            raise TypeError(f"audio_data must be bytes or numpy array, got {type(audio_data)}")
            
        # Clamp values
        if audio_array.max() > 1.0 or audio_array.min() < -1.0:
            audio_array = np.clip(audio_array, -1.0, 1.0)
        return audio_array

    def _run_diarization_model(self, audio_array, sample_rate=16000):
        """
        Run the diarization model.
        """
        if not self.diarize_model:
            return None
            
        try:
            # Pyannote expects a tensor (channels, time)
            if len(audio_array.shape) == 1:
                waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_array).float()
                
            if self.device:
                 dev = torch.device(self.device)
                 waveform = waveform.to(dev)
                 
            return self.diarize_model({"waveform": waveform, "sample_rate": sample_rate})
                
        except Exception as e:
            print(f"Diarization inference failed: {e}")
            return None

    def _assign_speakers(self, segments, audio_array, sample_rate=16000):
        """
        Manually assign speakers to segments using pyannote.
        """
        if not self.diarize_model:
            return segments
            
        try:
            # Run diarization
            diarization = self._run_diarization_model(audio_array, sample_rate)
            
            if diarization is None:
                return segments
            
            # Handle Pyannote outputs (Annotation or wrapped in DiarizeOutput/tuple)
            # Case 0: Pyannote DiarizeOutput wrapper (new in 3.1+)
            if hasattr(diarization, "annotation"):
                diarization = diarization.annotation
            elif hasattr(diarization, "speaker_diarization"):
                diarization = diarization.speaker_diarization
            
            # Case 0.5: Tuple/List wrapper
            elif isinstance(diarization, (tuple, list)) and len(diarization) > 0:
                if hasattr(diarization[0], "itertracks"):
                    diarization = diarization[0]

            # Convert to list of turns
            speaker_turns = []
            if hasattr(diarization, "itertracks"):
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_turns.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
            else:
                # Should not happen if correctly unpacked, but fallback
                print(f"Warning: Unknown diarization output format: {type(diarization)}")
            
            # Assign speakers to segments based on overlap
            for segment in segments:
                # Find overlapping speaker turns
                seg_start = segment['start']
                seg_end = segment['end']
                
                overlaps = []
                for turn in speaker_turns:
                    # Calculate overlap
                    overlap_start = max(seg_start, turn['start'])
                    overlap_end = min(seg_end, turn['end'])
                    overlap_dur = max(0, overlap_end - overlap_start)
                    
                    if overlap_dur > 0:
                        overlaps.append((overlap_dur, turn['speaker']))
                
                # Assign dominant speaker
                if overlaps:
                    # Sort by overlap duration desc
                    overlaps.sort(key=lambda x: x[0], reverse=True)
                    segment['speaker'] = overlaps[0][1]
                    
        except Exception as e:
            print(f"Diarization assignment failed: {e}")
            import traceback
            traceback.print_exc()
            
        return segments

    def transcribe_full_audio(self, audio_data, sample_rate: int = 16000) -> Iterator[dict]:
        """
        Transcribe full audio and yield segments with timestamps.
        """
        audio_array = self._prepare_audio(audio_data)
            
        try:
            segments = []
            
            if self.backend == "faster_whisper":
                # faster-whisper
                # Disable VAD filter for now
                fw_segments, _ = self.model.transcribe(
                    audio_array,
                    language=self.language,
                    beam_size=5,
                    vad_filter=False
                )
                
                # Convert format
                for seg in fw_segments:
                    segments.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                        "speaker": None
                    })
            
            else: # huggingface
                # Enable return_timestamps for full audio
                result = self.pipe(audio_array, return_timestamps=True)
                
                # Result format: {'text': '...', 'chunks': [{'timestamp': (0.0, 4.0), 'text': '...'}]}
                if "chunks" in result:
                    for chunk in result["chunks"]:
                        start, end = chunk.get("timestamp", (0.0, 0.0))
                        # Handle case where timestamp might be None or single float (rare but possible in old versions)
                        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                            segments.append({
                                "start": start,
                                "end": end,
                                "text": chunk.get("text", "").strip(),
                                "speaker": None
                            })
                else:
                    # Fallback if no chunks
                     segments.append({
                        "start": 0,
                        "end": len(audio_array) / sample_rate, 
                        "text": result.get("text", "").strip(),
                        "speaker": None
                    })
            
            # Apply diarization if enabled (Common to both backends)
            if self.enable_diarization and self.diarize_model:
                print("Running diarization...")
                segments = self._assign_speakers(segments, audio_array, sample_rate)
            
            for segment in segments:
                yield segment
                
        except Exception as e:
            print(f"Full transcription failed: {e}")
            # Yield an error segment so UI knows something went wrong
            yield {
                "start": 0.0,
                "end": 0.0,
                "text": f"Error: Transcription failed ({str(e)})",
                "speaker": "System"
            }

    def start_live_transcription(self, audio_capture, callback=None):
        """
        Start the live transcription thread.
        """
        if self.is_transcribing:
            print("Already transcribing!")
            return
        
        self.is_transcribing = True
        self.full_transcript = []
        
        self.transcription_thread = threading.Thread(
            target=self._transcription_loop,
            args=(audio_capture, callback),
            daemon=True
        )
        self.transcription_thread.start()
        print("Live transcription started!")

    def _transcription_loop(self, audio_capture, callback):
        """
        Main loop for processing audio chunks with overlap and deduplication.
        """
        # Buffer for audio context (keep last 30s)
        audio_buffer = AudioBuffer(max_duration=30.0, sample_rate=audio_capture.sample_rate)
        
        # Sliding window parameters
        context_duration = 10.0 # Transcribe last 10s to capture full phrases
        
        chunk_count = 0
        last_confirmed_text = ""
        
        while self.is_transcribing:
            # Get chunk from capture
            audio_chunk = audio_capture.get_audio_chunk(timeout=1.0)
            if audio_chunk is None:
                continue
                
            chunk_count += 1
            try:
                # Add to buffer
                audio_buffer.add_chunk(audio_chunk)
                
                start_time = time.time()
                
                # Get last N seconds for transcription context
                # We need enough context for Whisper to not hallucinate, but short enough for speed
                full_audio = audio_buffer.get_audio()
                
                # Calculate samples to keep for context
                samples_to_keep = int(context_duration * audio_capture.sample_rate)
                if len(full_audio) > samples_to_keep:
                    audio_window = full_audio[-samples_to_keep:]
                else:
                    audio_window = full_audio
                
                # Transcribe the window
                raw_text = self.transcribe_chunk(audio_window, audio_capture.sample_rate)
                
                if raw_text:
                    # Deduplicate against previous context
                    new_content = self._deduplicate_text(last_confirmed_text, raw_text)
                    
                    if new_content:
                        elapsed = time.time() - start_time
                        timestamp = time.strftime("%H:%M:%S")
                        
                        transcription_entry = {
                            "timestamp": timestamp,
                            "text": new_content,
                            "chunk": chunk_count
                        }
                        
                        self.full_transcript.append(transcription_entry)
                        self.transcript_queue.put(transcription_entry)
                        print(f"[{timestamp}] {new_content} (processed in {elapsed:.2f}s)")
                        
                        if callback:
                            callback(transcription_entry)
                        
                        # Update last confirmed text for next iteration
                        # Keep a reasonable amount of history for overlap checking
                        last_confirmed_text += " " + new_content
                        if len(last_confirmed_text) > 200:
                            last_confirmed_text = last_confirmed_text[-200:]
                        
            except Exception as e:
                print(f"Error transcribing chunk {chunk_count}: {e}")

    def _deduplicate_text(self, previous_text: str, current_text: str) -> str:
        """
        Remove overlap between previous text and current transcription.
        Simple string matching approach.
        """
        if not previous_text:
            return current_text
            
        previous_text = previous_text.strip()
        current_text = current_text.strip()
        
        if not current_text:
            return ""
            
        # 1. Exact match check (if the model just output exactly what it said before)
        if current_text in previous_text:
            return ""
            
        # 2. Overlap check
        # We look for the longest suffix of 'previous_text' that is a prefix of 'current_text'
        # Heuristic: verify at least 2 words or 10 chars match to avoid false positives on "the", "a", etc.
        
        # Normalize for comparison (lowercase, remove punctuation roughly)
        def normalize(s):
            return "".join(c.lower() for c in s if c.isalnum() or c.isspace()).strip()
            
        n_prev = normalize(previous_text)
        n_curr = normalize(current_text)
        
        # Try to find overlap
        overlap_len = 0
        max_overlap_check = min(len(n_prev), len(n_curr))
        
        # Check suffixes of prev against prefixes of curr
        for i in range(1, max_overlap_check + 1):
            suffix = n_prev[-i:]
            prefix = n_curr[:i]
            if suffix == prefix:
                overlap_len = i
        
        # If significant overlap found in normalized string
        if overlap_len > 10: # arbitrary threshold for "significant"
             # Find where this corresponds in the original current_text
             # This is tricky because we normalized. 
             # Fallback: aggressive string finding in raw text
             pass

        # Simpler word-based approach
        prev_words = previous_text.split()
        curr_words = current_text.split()
        
        if not prev_words or not curr_words:
            return current_text
            
        # Look for overlap in words
        overlap_index = 0
        min_overlap_words = 2
        
        # Check from maximum possible overlap down to minimum
        max_overlap = min(len(prev_words), len(curr_words))
        
        for i in range(max_overlap, 0, -1):
            # chunks of i words
            suffix = prev_words[-i:]
            prefix = curr_words[:i]
            
            # Simple list equality
            if suffix == prefix:
                # Basic match found
                if i >= min_overlap_words:
                    overlap_index = i
                    break
                else:
                    # For very short overlaps (1 word), requires context or strict checking
                    # Here we might skip to avoid eating "The" if sentences start with "The"
                    pass
        
        if overlap_index > 0:
            # Return only the new part
            new_words = curr_words[overlap_index:]
            if not new_words:
                return ""
            return " ".join(new_words)
            
        return current_text

    def stop_transcription(self):
        """Stop the transcription thread."""
        if not self.is_transcribing:
            return
        self.is_transcribing = False
        if self.transcription_thread:
            self.transcription_thread.join(timeout=5.0)
        print("Live transcription stopped.")

    def get_full_transcript(self) -> str:
        """Get the full transcript as a string."""
        lines = []
        for entry in self.full_transcript:
            lines.append(f"[{entry['timestamp']}] {entry['text']}")
        return "\n".join(lines)

    def get_transcript_entries(self) -> list[dict]:
        """Get list of transcript entries."""
        return self.full_transcript.copy()

    def clear_transcript(self):
        """Clear the current transcript."""
        self.full_transcript = []
        while not self.transcript_queue.empty():
            try:
                self.transcript_queue.get_nowait()
            except queue.Empty:
                break

