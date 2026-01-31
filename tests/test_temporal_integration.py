
import os
import sys
import shutil
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from src.temporal_graph_memory import TemporalGraphMemory, GraphNode, NodeType
from src.text_analysis import TextAnalyzer, RoleBasedHighlightScorer

def test_temporal_integration():
    print("Initializing components...")
    
    # Initialize components
    # Using CPU for test to be safe/fast
    text_analyzer = TextAnalyzer(device="cpu")
    
    # Create a fresh memory instance
    test_storage = "test_memory_storage"
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)
        
    temporal_memory = TemporalGraphMemory(
        storage_path=test_storage,
        similarity_threshold=0.5,
        text_analyzer=text_analyzer
    )
    
    # Create scorer WITH memory
    scorer = RoleBasedHighlightScorer(
        text_analyzer=text_analyzer,
        temporal_memory=temporal_memory
    )
    
    # Create scorer WITHOUT memory (control)
    control_scorer = RoleBasedHighlightScorer(
        text_analyzer=text_analyzer,
        temporal_memory=None
    )
    
    print("\n--- Setting up Historical Context ---")
    # Add a historical decision about "Python"
    # We want to see if future sentences about Python get boosted
    decision_text = "We have decided to use Python as our primary backend language."
    temporal_memory.add_decision(
        meeting_id="old_meeting",
        decision_text=decision_text,
        participants=["Alice", "Bob"]
    )
    print(f"Added past decision: '{decision_text}'")
    
    print("\n--- Testing Scoring ---")
    
    # Test Sentence 1: Relevant to history
    # This is somewhat generic on its own, but relevant to the decision
    test_sentence = "I think sticking with Python is the right choice."
    
    # Score with control (no memory)
    score_no_mem = control_scorer.score_sentence(test_sentence)
    
    # Score with memory
    score_mem = scorer.score_sentence(test_sentence)
    
    print(f"Test Sentence: '{test_sentence}'")
    print(f"Score (No Memory): {score_no_mem:.4f}")
    print(f"Score (With Memory): {score_mem:.4f}")
    
    boost = score_mem - score_no_mem
    print(f"Observed Boost: {boost:.4f}")
    
    if score_mem > score_no_mem:
        print("✅ SUCCESS: Temporal memory boosted the relevant sentence!")
    else:
        print("❌ FAILURE: No boost observed.")
        
    # Test Sentence 2: Irrelevant
    irrelevant_sentence = "The weather outside is very sunny today."
    score_irr_mem = scorer.score_sentence(irrelevant_sentence)
    print(f"\nIrrelevant Sentence: '{irrelevant_sentence}'")
    print(f"Score: {score_irr_mem:.4f}")
    
    # Verify irrelevant sentence didn't get a massive boost (it should be near 0)
    if score_irr_mem < 0.2:
        print("✅ SUCCESS: Irrelevant sentence has low score.")
    else:
        print(f"⚠️ WARNING: Irrelevant sentence score is high ({score_irr_mem:.4f}).")

    # Clean up
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)

if __name__ == "__main__":
    try:
        test_temporal_integration()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
