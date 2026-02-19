
import os
import sys
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from live_transcription import LiveTranscriber

def verify_mac_whisper():
    print("Verifying Whisper on Mac (MPS)...")
    
    try:
        # Initialize with Hugging Face pipeline on MPS
        print("Initializing Transcriber with model_type='huggingface' and device='mps'...")
        transcriber = LiveTranscriber(
            model_size="tiny", # Use tiny for speed
            device="mps",
            enable_diarization=True
        )
        
        # Create dummy audio (1 second of silence/noise)
        # 16000 sample rate, 1 second
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Generate a sine wave at 440 Hz
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio = audio.astype(np.float32)
        
        print("Transcribing dummy audio (full mode)...")
        start_time = time.time()
        result = list(transcriber.transcribe_full_audio(audio, sample_rate))
        end_time = time.time()
        
        print(f"Transcription result (segments): '{result}'")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        
        if isinstance(result, str):
            print("SUCCESS: Transcription returned a string.")
        else:
            print(f"FAILURE: Transcription returned {type(result)}.")
            
    except Exception as e:
        print(f"FAILURE: An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_mac_whisper()
