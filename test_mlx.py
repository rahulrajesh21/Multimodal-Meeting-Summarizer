import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.live_transcription import LiveTranscriber

transcriber = LiveTranscriber(
    model_size="base",
    device="mps",
    enable_diarization=False,
    backend="whisply"
)

# Dummy audio 16kHz
sample_rate = 16000
audio_bytes = (np.random.rand(sample_rate * 3) * 32767).astype(np.int16).tobytes()

for segment in transcriber.transcribe_full_audio(audio_bytes, sample_rate):
    print(segment)
