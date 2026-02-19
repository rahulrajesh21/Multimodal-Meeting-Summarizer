
import os
import sys
import numpy as np
import torch
from dotenv import load_dotenv
from src.live_transcription import LiveTranscriber
from src.audio_analysis import load_audio_file

load_dotenv()

def test_hf_diarization():
    print("Testing Hugging Face Diarization...")
    
    # Check for HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable not set.")
        return

    # Initialize transcriber
    try:
        transcriber = LiveTranscriber(
            model_size="tiny", # Use tiny for speed
            device="cpu",
            enable_diarization=True
        )
    except Exception as e:
        print(f"Failed to initialize transcriber: {e}")
        return

    # Check if diarization model loaded
    if not transcriber.diarize_model:
        print("Error: Diarization model failed to load.")
        return
    else:
        print("Diarization model loaded successfully.")

    # Create a dummy audio signal (speech-like)
    # We ideally need a real audio file with speakers.
    # For now, let's try to load a sample or create synthetic noise?
    # Synthetic noise won't work for diarization (needs speech).
    # Let's check if there are any audio files in the repo or create a dummy one.
    
    # Creating a dummy "silence" + "tone" is likely not enough for pyannote.
    # We need a file.
    # Let's see if we can use a sample file if it exists, otherwise warn.
    sample_audio = "sample_meeting.wav"
    if not os.path.exists(sample_audio):
        # Create a simple sine wave file just to test the pipeline doesn't crash
        # (It won't detect speakers but it will run the code)
        import wave
        with wave.open(sample_audio, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # Generate 5 seconds of "speech" (just random noise + sine to valid audio)
            t = np.linspace(0, 5, 16000*5)
            audio = np.sin(2 * np.pi * 440 * t) * 0.5
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        print(f"Created dummy audio file: {sample_audio}")

    # Load audio
    import librosa
    audio, sr = librosa.load(sample_audio, sr=16000)
    
    print(f"Transcribing full audio ({len(audio)/16000:.2f}s)...")
    
    # Run full transcription (which includes diarization)
    segments = list(transcriber.transcribe_full_audio(audio))
    
    print("\nResults:")
    for seg in segments:
        print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg.get('speaker', 'NoSpeaker')}: {seg['text']}")

    if not segments:
        print("No segments found.")

if __name__ == "__main__":
    test_hf_diarization()
