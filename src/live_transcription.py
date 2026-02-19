"""
Real-time transcription engine using Whisper for live audio streaming.
Supports both OpenAI Whisper API and local Whisper models.
"""

import os
import sys
import tempfile
import threading
import queue
from typing import Optional, Callable, Literal, Iterator
import time
import numpy as np

# Import transformers for MPS support
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
# Fix for PyTorch 2.6+ weights_only=True security change
# MUST be applied before importing whisperx or pyannote
import torch
from functools import partial

if hasattr(torch, 'load'):
    _original_load = torch.load
    
    def _safe_load(*args, **kwargs):
        # FORCE weights_only=False to support legacy checkpoints (WhisperX/Pyannote)
        # even if the caller explicitly requested weights_only=True
        if 'weights_only' in kwargs:
            del kwargs['weights_only']
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
        
    torch.load = _safe_load
    # print("Applied torch.load monkey-patch for weights_only=False")

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import whisper (faster-whisper for real-time performance)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not available. Install with: pip install faster-whisper")

# Try to import OpenAI for API-based transcription
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import pyannote.audio directly for robust diarization
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Install with: pip install pyannote.audio")

# Try to import whisperx
try:
    import whisperx
    from whisperx.diarize import DiarizationPipeline
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("Warning: whisperx not available. Install with: pip install whisperx")



from audio_capture import AudioCapture, AudioBuffer


class LiveTranscriber:
    """
    Real-time transcription engine that processes live audio streams.
    """
    
    def __init__(
        self,
        model_type: Literal["faster-whisper", "openai", "whisperx", "huggingface"] = "faster-whisper",
        model_size: str = "base",
        language: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        enable_diarization: bool = False
    ):
        """
        Initialize the live transcriber.
        
        Args:
            model_type: Type of model to use ("faster-whisper", "openai", "whisperx", or "huggingface")
            model_size: Model size for faster-whisper/whisperx (tiny, base, small, medium, large-v2, large-v3)
            language: Optional language code (e.g., "en", "es", "fr")
            openai_api_key: OpenAI API key (required if using openai model)
            hf_token: Hugging Face token (required for diarization)
            device: Device to run on ("cpu" or "cuda")
            compute_type: Compute type for faster-whisper/whisperx ("int8", "float16", "float32")
            enable_diarization: Enable speaker diarization
        """
        self.model_type = model_type
        self.model_size = model_size
        self.language = language
        self.device = device
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token
        self.diarize_model = None
        
        # Validate device compatibility
        if device == "mps" and model_type == "faster-whisper":
            raise ValueError(
                "faster-whisper does not support MPS (Apple Metal).\n\n"
                "To use Apple Silicon GPU acceleration:\n"
                "1. Select 'whisperx' as the Model Type\n"
                "2. Select 'mps' as the Device\n\n"
                "Alternatively, use 'cpu' device with faster-whisper."
            )
        
        # Initialize model based on type
        if model_type == "whisperx":
            if not WHISPERX_AVAILABLE:
                raise ImportError(
                    "WhisperX is not installed.\n\n"
                    "To install WhisperX:\n"
                    "1. Activate your virtual environment\n"
                    "2. Run: pip install whisperx\n"
                    "3. For GPU support: Install CUDA 12.8\n"
                    "4. For speaker diarization: Set HF_TOKEN environment variable\n\n"
                    "See WHISPERX_SETUP.md for detailed instructions."
                )
            
            print(f"Loading WhisperX model: {model_size}...")
            self.model = whisperx.load_model(
                model_size,
                device=device,
                compute_type=compute_type
            )
            print("WhisperX model loaded successfully!")
            
        elif model_type == "faster-whisper":
            if not FASTER_WHISPER_AVAILABLE:
                raise ImportError("faster-whisper is not installed. Install with: pip install faster-whisper")
            
            print(f"Loading faster-whisper model: {model_size}...")
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            print("Model loaded successfully!")
            
        elif model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package is not installed. Install with: pip install openai")
            
            if not openai_api_key:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
            
            self.client = OpenAI(api_key=openai_api_key)
            print("OpenAI client initialized!")
            
        elif model_type == "huggingface":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers is not installed. Install with: pip install transformers torch")
                
            print(f"Loading Hugging Face Whisper model: openai/whisper-{model_size} on {device}...")
            
            # Map device string to torch device
            if device == "cuda":
                device_arg = "cuda:0"
            elif device == "mps":
                device_arg = "mps"
            else:
                device_arg = "cpu"
                
            try:
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=f"openai/whisper-{model_size}",
                    chunk_length_s=30,
                    device=device_arg,
                )
                print(f"Hugging Face model loaded successfully on {device_arg}!")
            except Exception as e:
                raise RuntimeError(f"Failed to load Hugging Face pipeline: {e}")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Initialize diarization if enabled (for ANY model type that supports it)
        if enable_diarization:
            self._init_diarization(hf_token, device)
        
        # Transcription state
        self.is_transcribing = False
        self.transcription_thread: Optional[threading.Thread] = None
        self.transcript_queue = queue.Queue()
        self.full_transcript = []

    def _init_diarization(self, hf_token: Optional[str], device: str):
        """Initialize the diarization pipeline."""
        if not hf_token:
            hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            print("Warning: Diarization requires HF_TOKEN. Set environment variable or pass hf_token parameter.")
            self.enable_diarization = False
            return

        print("Initializing speaker diarization pipeline...")
        
        # Flag to track if we successfully loaded a model
        model_loaded = False
        self.diarization_backend = None  # Track which backend is used: "whisperx" or "pyannote"
        
        # 1. Try loading via WhisperX (wrapper around pyannote)
        if WHISPERX_AVAILABLE:
            try:
                print(f"Attempting to load WhisperX DiarizationPipeline with token: {hf_token[:4]}...")
                self.diarize_model = DiarizationPipeline(token=hf_token, device=device)
                print(f"DiarizationPipeline returned: {type(self.diarize_model)}")
                print("Diarization pipeline loaded (via WhisperX)!")
                model_loaded = True
                self.diarization_backend = "whisperx"
            except Exception as e:
                print(f"Warning: Failed to load WhisperX diarization: {e}")
                print("Falling back to direct pyannote.audio...")
        
        # 2. Fallback to direct pyannote.audio if WhisperX failed or not available
        if not model_loaded and PYANNOTE_AVAILABLE:
            try:
                print("Attempting to load pyannote/speaker-diarization-3.1 directly...")
                self.diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
                if self.diarize_model:
                    self.diarize_model.to(torch.device(device))
                    print("Diarization pipeline loaded (via pyannote)!")
                    model_loaded = True
                    self.diarization_backend = "pyannote"
                else:
                    print("Failed to load pyannote pipeline (returned None). Check HF_TOKEN and access rights.")
            except Exception as e:
                print(f"Error loading pyannote pipeline: {e}")
        
        if not model_loaded:
            print("Could not load any diarization pipeline.")
            self.enable_diarization = False
        
    def transcribe_chunk(self, audio_data, sample_rate: int = 16000) -> str:
        """
        Transcribe a single audio chunk.
        """
        if self.model_type == "whisperx":
            return self._transcribe_with_whisperx(audio_data, sample_rate)
        elif self.model_type == "faster-whisper":
            return self._transcribe_with_faster_whisper(audio_data, sample_rate)
        elif self.model_type == "openai":
            return self._transcribe_with_openai(audio_data, sample_rate)
        elif self.model_type == "huggingface":
            return self._transcribe_with_huggingface(audio_data, sample_rate)
    
    def _transcribe_with_whisperx(self, audio_data, sample_rate: int) -> str:
        """Transcribe using WhisperX."""
        # ... (existing implementation) ...
        # Convert to float32 array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.float32:
                audio_array = audio_data.astype(np.float32)
            else:
                audio_array = audio_data
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = np.clip(audio_array, -1.0, 1.0)
        else:
            raise TypeError(f"audio_data must be bytes or numpy array, got {type(audio_data)}")
        
        # Transcribe with WhisperX
        result = self.model.transcribe(
            audio_array,
            batch_size=16,
            language=self.language
        )
        
        # Apply alignment
        if result["segments"]:
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result.get("language", self.language or "en"),
                    device=self.device
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio_array,
                    self.device,
                    return_char_alignments=False
                )
            except Exception as e:
                print(f"Warning: Alignment failed: {e}")
        
        # Apply diarization if enabled
        if self.enable_diarization and self.diarize_model and result["segments"]:
            try:
                # Use helper to ensure correct input format for backend
                diarize_segments = self._run_diarization_model(audio_array, sample_rate)
                if diarize_segments is not None:
                     result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                print(f"Warning: Diarization failed: {e}")
        
        # Format output text
        text_parts = []
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "")
            text = segment.get("text", "").strip()
            if speaker:
                text_parts.append(f"[{speaker}] {text}")
            else:
                text_parts.append(text)
        
        return " ".join(text_parts).strip()
    
    def _transcribe_with_faster_whisper(self, audio_data, sample_rate: int) -> str:
        """Transcribe using faster-whisper."""
        # ... (existing implementation) ...
        # Convert to float32 array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.float32:
                audio_array = audio_data.astype(np.float32)
            else:
                audio_array = audio_data
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = np.clip(audio_array, -1.0, 1.0)
        else:
            raise TypeError(f"audio_data must be bytes or numpy array, got {type(audio_data)}")
        
        # Transcribe
        segments, info = self.model.transcribe(
            audio_array,
            language=self.language,
            beam_size=5,
            vad_filter=False  # Disabled - requires onnxruntime
        )
        
        # Note: We don't do diarization for chunks in real-time usually as it's slow
        # But if we wanted to, we could call _assign_speakers here.
        # For now, just return text.
        
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    
    def _transcribe_with_openai(self, audio_data: bytes, sample_rate: int) -> str:
        """Transcribe using OpenAI Whisper API."""
        # ... (existing implementation) ...
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            import wave
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
        
        try:
            with open(temp_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=self.language
                )
            return transcript.text.strip()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _transcribe_with_huggingface(self, audio_data, sample_rate: int) -> str:
        """Transcribe using Hugging Face transformers pipeline."""
        # Convert to float32 array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.float32:
                audio_array = audio_data.astype(np.float32)
            else:
                audio_array = audio_data
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = np.clip(audio_array, -1.0, 1.0)
        else:
            raise TypeError(f"audio_data must be bytes or numpy array, got {type(audio_data)}")

        # Helper method for the pipeline
        def _get_samples():
            yield audio_array

        try:
            # We can pass the numpy array directly to the pipeline?
            # The documentation says it accepts "numpy arrays", "str" (paths), or "bytes"
            # However, for live streaming chunks, we might want to be careful about format.
            # Usually strict requirement is sample rate matching the model (16kHz).
            
            # Since we are passing raw samples, we should probably wrap it in a dict with sampling_rate
            # actually the pipeline is flexible.
            
            prediction = self.pipe(audio_array, return_timestamps=False)
            
            if isinstance(prediction, dict):
                return prediction.get("text", "").strip()
            elif isinstance(prediction, list):
                # Should not happen with return_timestamps=False and short chunk
                 return " ".join([p.get("text", "") for p in prediction]).strip()
            else:
                 return str(prediction).strip()

        except Exception as e:
            print(f"Hugging Face transcription failed: {e}")
            return ""

    def _run_diarization_model(self, audio_array, sample_rate=16000):
        """
        Run the configured diarization model with correct input format.
        """
        if not self.diarize_model:
            return None
            
        try:
            # WhisperX backend -> expects numpy array
            if self.diarization_backend == "whisperx":
                return self.diarize_model(audio_array)
                
            # Pyannote backend -> expects torch tensor dict
            elif self.diarization_backend == "pyannote":
                # Pyannote expects a tensor (channels, time)
                # audio_array is typically mono (time,)
                if len(audio_array.shape) == 1:
                    waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
                else:
                    waveform = torch.from_numpy(audio_array).float()
                    
                if self.device:
                     # Map 'cuda' or 'mps' or 'cpu' correctly
                     # If device is 'cuda:0', torch.device handles it
                     dev = torch.device(self.device)
                     waveform = waveform.to(dev)
                     
                return self.diarize_model({"waveform": waveform, "sample_rate": sample_rate})
            
            # Fallback/Unknown -> try WhisperX style first as it is simpler
            else:
                return self.diarize_model(audio_array)
                
        except Exception as e:
            print(f"Diarization inference failed: {e}")
            return None

    def _assign_speakers(self, segments, audio_array, sample_rate=16000):
        """
        Manually assign speakers to faster-whisper/HF segments using pyannote.
        """
        if not self.diarize_model:
            return segments
            
        try:
            # Run diarization using unified helper
            diarization = self._run_diarization_model(audio_array, sample_rate)
            
            if diarization is None:
                return segments
            
            # Convert diarization to list of turns
            speaker_turns = []
            
            # Case 0: Pyannote DiarizeOutput wrapper (new in 3.1+)
            if hasattr(diarization, "annotation"):
                diarization = diarization.annotation
            elif hasattr(diarization, "speaker_diarization"):
                diarization = diarization.speaker_diarization
            
            # Case 0.5: Tuple/List wrapper (some versions/hooks return (annotation, ...))
            elif isinstance(diarization, (tuple, list)) and len(diarization) > 0:
                if hasattr(diarization[0], "itertracks"):
                    diarization = diarization[0]

            # Case 1: Pyannote Annotation object (direct usage)
            if hasattr(diarization, "itertracks"):
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_turns.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
            # Case 2: Pandas DataFrame (WhisperX usage)
            elif hasattr(diarization, "itertuples"):
                for row in diarization.itertuples():
                    speaker_turns.append({
                        "start": row.start,
                        "end": row.end,
                        "speaker": row.speaker
                    })
            # Case 3: Dictionary (Generic fallback)
            elif isinstance(diarization, dict) and "segments" in diarization:
                 for seg in diarization["segments"]:
                    speaker_turns.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker": seg["speaker"]
                    })
            else:
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
        # Convert to float32 array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.float32:
                audio_array = audio_data.astype(np.float32)
            else:
                audio_array = audio_data
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = np.clip(audio_array, -1.0, 1.0)
        else:
            raise TypeError(f"audio_data must be bytes or numpy array, got {type(audio_data)}")
            
        if self.model_type == "whisperx":
            # ... (existing whisperx logic) ...
            result = self.model.transcribe(
                audio_array,
                batch_size=16,
                language=self.language
            )
            
            if result["segments"]:
                try:
                    model_a, metadata = whisperx.load_align_model(
                        language_code=result.get("language", self.language or "en"),
                        device=self.device
                    )
                    result = whisperx.align(
                        result["segments"],
                        model_a,
                        metadata,
                        audio_array,
                        self.device,
                        return_char_alignments=False
                    )
                except Exception as e:
                    print(f"Warning: Alignment failed: {e}")
            
            if self.enable_diarization and self.diarize_model and result["segments"]:
                try:
                    diarize_segments = self.diarize_model(audio_array)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                except Exception as e:
                    print(f"Warning: Diarization failed: {e}")
            
            for segment in result.get("segments", []):
                yield {
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "").strip(),
                    "speaker": segment.get("speaker", None)
                }
                
        elif self.model_type == "faster-whisper":
            # Transcribe with faster-whisper
            segments_gen, info = self.model.transcribe(
                audio_array,
                language=self.language,
                beam_size=5,
                vad_filter=False  # Disabled - requires onnxruntime
            )
            
            # Convert generator to list for processing
            segments = []
            for s in segments_gen:
                segments.append({
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip(),
                    "speaker": None
                })
            
            # Apply diarization if enabled
            if self.enable_diarization and self.diarize_model:
                print("Running diarization on faster-whisper output...")
                segments = self._assign_speakers(segments, audio_array, sample_rate)
                
            for segment in segments:
                yield segment
                
        elif self.model_type == "openai":
            # ... (existing openai logic) ...
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                import wave
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())
            
            try:
                with open(temp_path, 'rb') as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=self.language,
                        response_format="verbose_json"
                    )
                
                if hasattr(response, 'segments'):
                    for segment in response.segments:
                        yield {
                            "start": segment['start'],
                            "end": segment['end'],
                            "text": segment['text'].strip(),
                            "speaker": None
                        }
                else:
                    yield {
                        "start": 0,
                        "end": response.duration,
                        "text": response.text.strip(),
                        "speaker": None
                    }
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        elif self.model_type == "huggingface":
            # Transcribe with Hugging Face pipeline
            # Convert audio properly
             # Convert to float32 array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, np.ndarray):
                 if audio_data.dtype != np.float32:
                    audio_array = audio_data.astype(np.float32)
                 else:
                    audio_array = audio_data
            
            try:
                # Enable return_timestamps for full audio
                result = self.pipe(audio_array, return_timestamps=True)
                
                # Result format: {'text': '...', 'chunks': [{'timestamp': (0.0, 4.0), 'text': '...'}]}
                segments = []
                if "chunks" in result:
                    for chunk in result["chunks"]:
                        start, end = chunk.get("timestamp", (0.0, 0.0))
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
                        "end": 0, 
                        "text": result.get("text", "").strip(),
                        "speaker": None
                    })
                
                # Apply diarization if enabled
                if self.enable_diarization and self.diarize_model:
                    print("Running diarization on Hugging Face output...")
                    segments = self._assign_speakers(segments, audio_array, sample_rate)
                
                for segment in segments:
                    yield segment
                    
            except Exception as e:
                print(f"Hugging Face full transcription failed: {e}")

    # ... (rest of the class methods: start_live_transcription, etc.) ...
    def start_live_transcription(self, audio_capture, callback=None):
        # ... (existing) ...
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
        # ... (existing) ...
        chunk_count = 0
        while self.is_transcribing:
            audio_chunk = audio_capture.get_audio_chunk(timeout=1.0)
            if audio_chunk is None:
                continue
            chunk_count += 1
            try:
                start_time = time.time()
                text = self.transcribe_chunk(audio_chunk, audio_capture.sample_rate)
                elapsed = time.time() - start_time
                if text:
                    timestamp = time.strftime("%H:%M:%S")
                    transcription_entry = {
                        "timestamp": timestamp,
                        "text": text,
                        "chunk": chunk_count
                    }
                    self.full_transcript.append(transcription_entry)
                    self.transcript_queue.put(transcription_entry)
                    print(f"[{timestamp}] {text} (processed in {elapsed:.2f}s)")
                    if callback:
                        callback(transcription_entry)
            except Exception as e:
                print(f"Error transcribing chunk {chunk_count}: {e}")

    def stop_transcription(self):
        # ... (existing) ...
        if not self.is_transcribing:
            return
        self.is_transcribing = False
        if self.transcription_thread:
            self.transcription_thread.join(timeout=5.0)
        print("Live transcription stopped.")

    def get_full_transcript(self) -> str:
        # ... (existing) ...
        lines = []
        for entry in self.full_transcript:
            lines.append(f"[{entry['timestamp']}] {entry['text']}")
        return "\n".join(lines)

    def get_transcript_entries(self) -> list[dict]:
        return self.full_transcript.copy()

    def clear_transcript(self):
        self.full_transcript = []
        while not self.transcript_queue.empty():
            try:
                self.transcript_queue.get_nowait()
            except queue.Empty:
                break



if __name__ == "__main__":
    # Test the live transcription
    print("Testing Live Transcription...")
    
    # Initialize audio capture
    capture = AudioCapture(chunk_duration=3.0)
    
    # Initialize transcriber (using faster-whisper for local testing)
    transcriber = LiveTranscriber(
        model_type="faster-whisper",
        model_size="base",
        language="en"
    )
    
    # Start recording
    capture.start_recording()
    
    # Start transcription
    transcriber.start_live_transcription(capture)
    
    print("\nSpeak into your microphone... (recording for 30 seconds)")
    time.sleep(30)
    
    # Stop everything
    transcriber.stop_transcription()
    capture.stop_recording()
    
    # Print full transcript
    print("\n" + "="*50)
    print("FULL TRANSCRIPT:")
    print("="*50)
    print(transcriber.get_full_transcript())
    
    # Cleanup
    capture.cleanup()
