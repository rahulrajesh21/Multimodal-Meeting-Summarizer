
import json
import os
import logging
from typing import Dict, Any, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackManager:
    """
    Manages user feedback and online learning for the Fusion Layer.
    
    Responsibilities:
    1. Persist and load fusion weights.
    2. Log user feedback (Likes/Dislikes).
    3. Update weights based on feedback (Online Learning).
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.weights_file = os.path.join(data_dir, "model_weights.json")
        self.feedback_log_file = os.path.join(data_dir, "feedback_log.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Default starting weights (matches FusionLayer defaults)
        self.default_weights = {
            'semantic': 0.4,
            'tonal': 0.15,
            'role': 0.25,
            'temporal': 0.2
        }
        
    def load_weights(self) -> Dict[str, float]:
        """Load weights from disk or return defaults."""
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    weights = json.load(f)
                return weights
            except Exception as e:
                logger.error(f"Error loading weights: {e}")
                return self.default_weights.copy()
        else:
            return self.default_weights.copy()
            
    def save_weights(self, weights: Dict[str, float]):
        """Save current weights to disk."""
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(weights, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
            
    def log_feedback(self, segment_text: str, scores: Dict[str, float], feedback_type: str):
        """
        Log user feedback for analysis.
        
        Args:
            segment_text: The text of the segment
            scores: Component scores (semantic, tonal, etc.)
            feedback_type: 'like' (1) or 'dislike' (-1)
        """
        entry = {
            'timestamp': time.time(),
            'text': segment_text[:50] + "...",
            'scores': scores,
            'feedback': feedback_type
        }
        
        try:
            # Append to log file (simple JSONL-like or list)
            existing_logs = []
            if os.path.exists(self.feedback_log_file):
                with open(self.feedback_log_file, 'r') as f:
                    try:
                        existing_logs = json.load(f)
                    except json.JSONDecodeError:
                        pass
            
            existing_logs.append(entry)
            
            # Keep last 1000 logs
            if len(existing_logs) > 1000:
                existing_logs = existing_logs[-1000:]
                
            with open(self.feedback_log_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")
            
    def update_weights(
        self,
        current_weights: Dict[str, float],
        segment_scores: Dict[str, float],
        feedback_value: float,  # +1.0 for Like, -1.0 for Dislike
        learning_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Adjust weights based on feedback (Online Learning).
        
        Logic:
        - If Like (+1): Increase weights of components that contributed heavily to the score.
        - If Dislike (-1): Decrease weights of components that contributed heavily (blame them).
        """
        new_weights = current_weights.copy()
        
        # Calculate contribution of each component
        # We assume the fused score was a weighted sum
        total_score = sum(current_weights[k] * segment_scores.get(k, 0.0) for k in current_weights)
        
        if total_score == 0:
            return new_weights
            
        # Update each weight
        # Rule: W_new = W_old + (LearningRate * Feedback * ComponentScore)
        # Detailed: If feedback is +1, we want to boost components with high scores.
        #           If feedback is -1, we want to penalize components with high scores.
        
        for key in new_weights:
            component_score = segment_scores.get(key, 0.0)
            
            # Gradient update
            delta = learning_rate * feedback_value * component_score
            
            # Apply update
            new_weights[key] += delta
            
            # Clip to ensure non-negative
            new_weights[key] = max(0.01, new_weights[key])
            
        # Normalize weights to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for key in new_weights:
                new_weights[key] /= total_weight
                
        # Persist changes
        self.save_weights(new_weights)
        
        return new_weights
