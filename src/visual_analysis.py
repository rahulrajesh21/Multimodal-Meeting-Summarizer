
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualAnalyzer:
    """
    Analyzer for extracting visual features and context from video frames.
    Implements:
    1. Fine-tuned ResNet-50 for Context Classification (Slide vs Person)
    2. Feature Extraction for Similarity (Embeddings)
    3. OCR for Text Extraction
    """
    
    def __init__(self, device: str = None, model_path: str = "resnet50_meeting_context.pth"):
        if device is None:
            from src.device_utils import get_best_device
            device = get_best_device()
        self.device = device
        self.model_path = model_path
        self.feature_extractor = None
        self.classifier = None
        self.transform = None
        self.ocr_reader = None
        self.is_ready = False
        
        # Initialize immediately
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize Fine-Tuned ResNet-50 and EasyOCR."""
        # 1. Try to load ResNet-50 (optional — heuristic works without it)
        try:
            logger.info(f"Initializing VisualAnalyzer on {self.device}...")
            
            from torchvision import models, transforms
            
            # Initialize standard ResNet-50 structure
            full_model = models.resnet50(pretrained=False) # We load custom weights
            
            # Modify the final layer to match our fine-tuning (2 classes)
            num_ftrs = full_model.fc.in_features
            full_model.fc = nn.Linear(num_ftrs, 2)
            
            # Load custom weights if available
            if os.path.exists(self.model_path):
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                state_dict = torch.load(self.model_path, map_location=self.device)
                full_model.load_state_dict(state_dict)
            else:
                logger.warning(f"Fine-tuned model not found at {self.model_path}. Using heuristic detection.")
            
            full_model.eval()
            
            # Handle device
            if self.device == "cuda" and torch.cuda.is_available():
                full_model = full_model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                full_model = full_model.to("mps")
                
            # Split into Feature Extractor and Classifier
            self.feature_extractor = nn.Sequential(*list(full_model.children())[:-1])
            self.classifier = full_model.fc
            
            # Define transform (standard ImageNet normalization)
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            self.is_ready = True
            logger.info("VisualAnalyzer CNN model initialized successfully.")
            
        except Exception as e:
            logger.warning(f"CNN model not available ({e}). Using heuristic screen-sharing detection.")
            self.is_ready = False

        # 2. Try to load EasyOCR (optional — works without it)
        try:
            import easyocr
            use_gpu = (self.device == "cuda")
            self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            logger.info("EasyOCR initialized successfully.")
        except Exception as e:
            logger.warning(f"EasyOCR not available ({e}). OCR text extraction disabled.")

    def extract_frames(self, video_path: str, interval: float = 2.0) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video at a given time interval."""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
            
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            return []
            
        frame_interval = int(fps * interval)
        current_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame % frame_interval == 0:
                timestamp = current_frame / fps
                frames.append((timestamp, frame))
                
            current_frame += 1
            
        cap.release()
        return frames

    def _detect_skin_ratio(self, frame_bgr: np.ndarray) -> float:
        """
        Detect the ratio of skin-tone pixels in a frame.
        Useful for distinguishing gallery views (many faces) from actual slides.
        """
        try:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            h, w = frame_bgr.shape[:2]
            # Skin tone range in HSV (covers diverse skin tones)
            lower_skin = np.array([0, 30, 60], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            return np.count_nonzero(skin_mask) / (h * w)
        except Exception:
            return 0.0

    def _detect_face_grid(self, frame_bgr: np.ndarray) -> bool:
        """
        Detect whether this looks like a video-conferencing gallery view
        by checking for a grid pattern of rectangular regions separated
        by thin dark borders (typical of Zoom/Teams gallery layouts).
        """
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Gallery borders are usually very dark (<30) or very bright (>225)
            dark_mask = (gray < 30).astype(np.uint8)

            # Check for horizontal dark bands spanning most of the width
            row_darkness = np.mean(dark_mask, axis=1)  # fraction of dark pixels per row
            h_bands = np.sum(row_darkness > 0.5)  # rows that are mostly dark

            # Check for vertical dark bands spanning most of the height
            col_darkness = np.mean(dark_mask, axis=0)
            v_bands = np.sum(col_darkness > 0.3)

            # A gallery view typically has 2-6 horizontal and 2-6 vertical dividers
            # spanning a significant portion of the frame
            has_grid = h_bands > (h * 0.02) and v_bands > (w * 0.02)
            return has_grid
        except Exception:
            return False

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str, float]:
        """
        Process a single frame to get:
        1. Visual Embedding (2048-dim)
        2. Context Label (Slide/Person)
        3. Context Confidence

        Uses a hybrid approach: CNN prediction is cross-checked with
        skin-tone detection to avoid misclassifying gallery views as slides.
        """
        if not self.is_ready or self.feature_extractor is None:
            return None, "Unknown", 0.0
            
        try:
            # Preprocess
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Move to device
            device = next(self.feature_extractor.parameters()).device
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                # 1. Get Features (Embedding)
                # Output shape: [1, 2048, 1, 1]
                features = self.feature_extractor(input_tensor)
                embedding = features.squeeze().cpu().numpy()
                
                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                # 2. Get Classification (Context)
                # Flatten features for fc layer: [1, 2048]
                flat_features = features.view(features.size(0), -1)
                logits = self.classifier(flat_features)
                probs = torch.softmax(logits, dim=1)
                
                # Classes: 0=Person, 1=Slide (Based on our training order)
                slide_prob = probs[0][1].item()
                person_prob = probs[0][0].item()
                
                if slide_prob > person_prob:
                    label = "Slide"
                    conf = slide_prob
                else:
                    label = "Person"
                    conf = person_prob

            # 3. Hybrid override: catch gallery views misclassified as slides
            # Real slides/screen shares have virtually no skin-tone pixels.
            # Gallery views (grid of webcam tiles) have many small faces.
            if label == "Slide":
                skin_ratio = self._detect_skin_ratio(frame_bgr)
                has_grid = self._detect_face_grid(frame_bgr)
                # If >3% skin and grid-like layout → gallery view, not a slide
                if skin_ratio > 0.03 and has_grid:
                    logger.debug(
                        f"Overriding CNN 'Slide' → 'Person' "
                        f"(skin={skin_ratio:.2%}, grid={has_grid})"
                    )
                    label = "Person"
                    conf = max(person_prob, 1.0 - conf)  # flip confidence
                # Even without grid, very high skin ratio means faces
                elif skin_ratio > 0.08:
                    logger.debug(
                        f"Overriding CNN 'Slide' → 'Person' "
                        f"(high skin={skin_ratio:.2%})"
                    )
                    label = "Person"
                    conf = max(person_prob, 1.0 - conf)

            return embedding, label, conf
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None, "Error", 0.0

    def extract_ocr_text(self, frame_bgr: np.ndarray) -> str:
        """Extract text from frame using EasyOCR."""
        if not self.is_ready or self.ocr_reader is None:
            return ""
            
        try:
            results = self.ocr_reader.readtext(frame_bgr, detail=0)
            return " ".join(results)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""

    def analyze_video_context(self, video_path: str) -> List[Dict]:
        """
        Full pipeline: Extract frames -> Get Embeddings -> Classify Context -> OCR.
        Falls back to heuristic detection if the trained model isn't available.
        """
        frames = self.extract_frames(video_path, interval=5.0) # Every 5 seconds
        results = []
        
        logger.info(f"Analyzing {len(frames)} frames from video...")
        
        for timestamp, frame in frames:
            if self.is_ready and self.feature_extractor is not None:
                # Full CNN pipeline
                embedding, label, conf = self.process_frame(frame)
                ocr_text = ""
                if label == "Slide" or conf < 0.7:
                    ocr_text = self.extract_ocr_text(frame)
                if embedding is not None:
                    results.append({
                        'timestamp': timestamp,
                        'embedding': embedding,
                        'context_label': label,
                        'context_confidence': conf,
                        'ocr_text': ocr_text,
                        'is_slide': (label == "Slide")
                    })
            else:
                # Heuristic fallback (no trained model needed)
                is_slide, conf = self.heuristic_detect_screen_share(frame)
                label = "Slide" if is_slide else "Person"
                ocr_text = ""
                if is_slide:
                    ocr_text = self.extract_ocr_text(frame) if self.ocr_reader else ""
                results.append({
                    'timestamp': timestamp,
                    'embedding': None,
                    'context_label': label,
                    'context_confidence': conf,
                    'ocr_text': ocr_text,
                    'is_slide': is_slide
                })
                
        slide_count = sum(1 for r in results if r['is_slide'])
        logger.info(f"Visual analysis complete: {len(results)} frames, {slide_count} screen-sharing detected")
        return results

    def heuristic_detect_screen_share(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Detect screen sharing using image heuristics (no ML model needed).
        
        Screen shares typically have:
        - High edge density (text, UI elements, code)
        - More uniform color regions (flat UI backgrounds)
        - Less skin-tone pixels
        - More straight horizontal/vertical lines
        
        Returns:
            Tuple of (is_screen_share: bool, confidence: float)
        """
        try:
            h, w = frame_bgr.shape[:2]
            
            # 1. Edge density — screen shares have lots of sharp edges (text, UI)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (h * w)
            
            # 2. Color uniformity — screen shares have large flat color regions
            # Divide into blocks and check color variance
            block_size = 32
            variances = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = frame_bgr[y:y+block_size, x:x+block_size]
                    variances.append(np.var(block))
            avg_variance = np.mean(variances) if variances else 100
            # Low variance blocks = flat colors (UI elements)
            low_var_ratio = sum(1 for v in variances if v < 200) / max(len(variances), 1)
            
            # 3. Skin tone detection — faces have skin-colored pixels
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            # Skin tone range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.count_nonzero(skin_mask) / (h * w)
            
            # 4. Straight line detection — screen shares have UI borders, text lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
            line_count = len(lines) if lines is not None else 0
            line_density = min(line_count / 100.0, 1.0)  # Normalize
            
            # Score: higher = more likely screen share
            score = (
                0.30 * min(edge_density * 10, 1.0) +   # Edge density (boosted)
                0.25 * low_var_ratio +                   # Flat color regions
                0.25 * (1.0 - min(skin_ratio * 5, 1.0)) + # Lack of skin tones
                0.20 * line_density                      # Straight lines
            )
            
            is_screen_share = score > 0.55
            return is_screen_share, float(score)
            
        except Exception as e:
            logger.error(f"Heuristic screen detection error: {e}")
            return False, 0.0

