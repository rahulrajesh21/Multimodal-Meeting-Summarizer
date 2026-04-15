"""
Auto-detect the best available PyTorch device (MPS > CUDA > CPU).
"""
import torch
import logging

logger = logging.getLogger(__name__)

def get_best_device() -> str:
    """Return the best available device string for PyTorch."""
    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS (Metal Performance Shaders) GPU")
        return "mps"
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return "cuda"
    logger.info("Using CPU")
    return "cpu"
