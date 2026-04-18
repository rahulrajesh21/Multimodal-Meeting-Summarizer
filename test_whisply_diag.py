"""Diagnostic: dump the exact structure returned by Whisply's transcribe_with_mlx_whisper"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.live_transcription import LiveTranscriber

transcriber = LiveTranscriber(
    model_size="base",
    device="mps",
    enable_diarization=False,
    backend="whisply"
)

# 3 seconds of random noise at 16kHz
sample_rate = 16000
audio_bytes = (np.random.rand(sample_rate * 3) * 32767).astype(np.int16).tobytes()

print("\n=== Collecting segments from transcribe_full_audio ===")
segments = list(transcriber.transcribe_full_audio(audio_bytes, sample_rate))
print(f"\nTotal segments returned: {len(segments)}")
for i, seg in enumerate(segments):
    print(f"  Segment {i}: {seg}")

# Also test the raw whisply output directly
print("\n=== Testing raw Whisply output ===")
import tempfile, soundfile as sf
from pathlib import Path

audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    sf.write(f.name, audio_array, sample_rate)
    audio_path = Path(f.name)

result = transcriber.whisply_handler.transcribe_with_mlx_whisper(audio_path)
os.remove(audio_path)

print(f"\nTop-level keys: {list(result.keys())}")
if "transcription" in result:
    t = result["transcription"]
    print(f"transcription keys: {list(t.keys())}")
    if "transcriptions" in t:
        ts = t["transcriptions"]
        print(f"transcriptions keys: {list(ts.keys())}")
        for lang, data in ts.items():
            print(f"\n  Language '{lang}' keys: {list(data.keys())}")
            if "segments" in data:
                segs = data["segments"]
                print(f"  segments count: {len(segs)}")
                if segs:
                    print(f"  First segment keys: {list(segs[0].keys())}")
                    print(f"  First segment: {segs[0]}")
            if "chunks" in data:
                chunks = data["chunks"]
                print(f"  chunks count: {len(chunks)}")
                if chunks:
                    print(f"  First chunk keys: {list(chunks[0].keys())}")
                    print(f"  First chunk: {chunks[0]}")
            # Print all keys with sample
            for k, v in data.items():
                if k not in ("segments", "chunks", "text"):
                    print(f"  '{k}': {type(v).__name__} = {str(v)[:200]}")
                elif k == "text":
                    print(f"  'text': {str(v)[:200]}")
