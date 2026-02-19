import time
import queue
import threading
from src.live_transcription import LiveTranscriber

def test_duplication_logic():
    print("Testing deduplication logic...")
    transcriber = LiveTranscriber(model_size="tiny", device="cpu")
    
    # Test 1: Exact overlap
    prev = "Hello world this is a test"
    curr = "world this is a test"
    result = transcriber._deduplicate_text(prev, curr)
    print(f"Test 1 (Exact partial): '{prev}' + '{curr}' -> '{result}'")
    assert result == "", f"Expected empty string, got '{result}'"
    
    # Test 2: Partial overlap with new content
    prev = "The quick brown fox"
    curr = "brown fox jumps over the lazy dog"
    result = transcriber._deduplicate_text(prev, curr)
    print(f"Test 2 (Overlap+New): '{prev}' + '{curr}' -> '{result}'")
    assert result == "jumps over the lazy dog", f"Expected 'jumps over the lazy dog', got '{result}'"
    
    # Test 3: No overlap
    prev = "First sentence."
    curr = "Second sentence."
    result = transcriber._deduplicate_text(prev, curr)
    print(f"Test 3 (No overlap): '{prev}' + '{curr}' -> '{result}'")
    assert result == "Second sentence.", f"Expected 'Second sentence.', got '{result}'"
    
    # Test 4: Short overlap (should be ignored if < 2 words)
    prev = "End of the line"
    curr = "Line of people"  # "Line" matches "line" (case insensitive ideally, but logic is case sensitive first)
    # Our logic uses split() so "line" != "Line"
    result = transcriber._deduplicate_text(prev, curr)
    print(f"Test 4 (Case mismatch/Short): '{prev}' + '{curr}' -> '{result}'")
    
    print("\nAll unit tests passed!")

if __name__ == "__main__":
    test_duplication_logic()
