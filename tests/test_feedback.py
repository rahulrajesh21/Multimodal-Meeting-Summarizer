
import os
import sys
import shutil
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from src.feedback_manager import FeedbackManager

def test_feedback_logic():
    print("Testing Feedback Manager...")
    
    # Setup test dir
    test_dir = "test_feedback_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    fm = FeedbackManager(data_dir=test_dir)
    
    # 1. Test Load Defaults
    weights = fm.load_weights()
    print(f"Initial Weights: {weights}")
    assert weights['semantic'] == 0.4
    
    # 2. Test Update (Like)
    # Scenario: High semantic score, user likes it -> Semantic weight should increase
    scores = {'semantic': 0.9, 'tonal': 0.1, 'role': 0.1, 'temporal': 0.1}
    print(f"\nFeedback: Like (+1) on segment with High Semantic score ({scores['semantic']})")
    
    new_weights = fm.update_weights(weights, scores, feedback_value=1.0, learning_rate=0.1)
    print(f"New Weights: {new_weights}")
    
    if new_weights['semantic'] > weights['semantic']:
        print("✅ SUCCESS: Semantic weight increased.")
    else:
        print("❌ FAILURE: Semantic weight did not increase.")
        
    # 3. Test Persistence
    fm2 = FeedbackManager(data_dir=test_dir)
    loaded_weights = fm2.load_weights()
    print(f"\nPersisted Weights: {loaded_weights}")
    assert abs(loaded_weights['semantic'] - new_weights['semantic']) < 0.001
    print("✅ SUCCESS: Weights persisted correctly.")
    
    # 4. Test Update (Dislike)
    # Scenario: High tonal score, user dislikes it -> Tonal weight should decrease
    scores = {'semantic': 0.1, 'tonal': 0.9, 'role': 0.1, 'temporal': 0.1}
    print(f"\nFeedback: Dislike (-1) on segment with High Tonal score ({scores['tonal']})")
    
    newer_weights = fm.update_weights(new_weights, scores, feedback_value=-1.0, learning_rate=0.1)
    print(f"Newer Weights: {newer_weights}")
    
    if newer_weights['tonal'] < new_weights['tonal']:
        print("✅ SUCCESS: Tonal weight decreased.")
    else:
        print("❌ FAILURE: Tonal weight did not decrease.")

    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_feedback_logic()
