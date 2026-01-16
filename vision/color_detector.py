"""
Color Detector for Science Hub
Detects color changes in test tubes using endoscopic camera
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ColorResult:
    """Result of color detection for a test tube"""
    tube_id: int
    test_name: str
    hex_color: str
    rgb: Tuple[int, int, int]
    hsv: Tuple[int, int, int]
    result: str
    confidence: float


class ColorDetector:
    """Detects dominant colors in test tube regions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tube_rois = config.get('tube_rois', [])
        self.test_mappings = config.get('test_mappings', {})
        self.camera = None
        self.camera_device = 0
    
    def initialize(self, camera_device: int = 0) -> bool:
        """Initialize camera"""
        import time
        self.camera_device = camera_device
        try:
            self.camera = cv2.VideoCapture(camera_device)
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {camera_device}")
                return False
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Camera warm-up: discard first few frames
            logger.info(f"Camera {camera_device} warming up...")
            time.sleep(1.0)  # Wait for camera to stabilize
            for _ in range(5):
                self.camera.read()  # Discard frames
            
            logger.info(f"Color detector initialized with camera {camera_device}")
            return True
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if not self.camera or not self.camera.isOpened():
            # Try to reinitialize
            if not self.initialize(self.camera_device):
                return None
        
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None
        return frame
    
    def detect_tube_colors(self, frame: np.ndarray, test_names: List[str]) -> List[ColorResult]:
        """Detect colors in all test tube regions"""
        results = []
        
        for i, roi in enumerate(self.tube_rois):
            if i >= len(test_names):
                break
            
            test_name = test_names[i]
            x, y, w, h = roi
            
            # Extract ROI
            tube_region = frame[y:y+h, x:x+w]
            
            if tube_region.size == 0:
                logger.warning(f"Empty ROI for tube {i+1}")
                continue
            
            # Get dominant color
            dominant_rgb = self._get_dominant_color(tube_region)
            dominant_hsv = self._rgb_to_hsv(dominant_rgb)
            hex_color = self._rgb_to_hex(dominant_rgb)
            
            # Classify result based on color
            result, confidence = self._classify_color(dominant_hsv, test_name)
            
            results.append(ColorResult(
                tube_id=i + 1,
                test_name=test_name,
                hex_color=hex_color,
                rgb=dominant_rgb,
                hsv=dominant_hsv,
                result=result,
                confidence=confidence
            ))
        
        return results
    
    def _get_dominant_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Get dominant color in image region using k-means"""
        # Reshape to list of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Use k-means to find dominant color
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find most common cluster
        _, counts = np.unique(labels, return_counts=True)
        dominant_idx = np.argmax(counts)
        dominant_color = centers[dominant_idx].astype(int)
        
        # OpenCV uses BGR, convert to RGB
        return (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))
    
    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert RGB to HSV"""
        # Create a 1x1 pixel image
        pixel = np.uint8([[list(rgb)[::-1]]])  # RGB to BGR
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        return (int(hsv[0][0][0]), int(hsv[0][0][1]), int(hsv[0][0][2]))
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex string"""
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    
    def _classify_color(self, hsv: Tuple[int, int, int], test_name: str) -> Tuple[str, float]:
        """Classify color based on test mappings"""
        if test_name not in self.test_mappings:
            return ("unknown", 0.0)
        
        mappings = self.test_mappings[test_name]
        h, s, v = hsv
        
        best_match = "unknown"
        best_confidence = 0.0
        
        for mapping in mappings:
            color_range = mapping.get('color_range', [])
            if len(color_range) != 2:
                continue
            
            lower = np.array(color_range[0])
            upper = np.array(color_range[1])
            hsv_arr = np.array([h, s, v])
            
            # Check if color falls within range
            if np.all(hsv_arr >= lower) and np.all(hsv_arr <= upper):
                # Calculate confidence based on how centered the color is
                center = (lower + upper) / 2
                distance = np.linalg.norm(hsv_arr - center)
                max_distance = np.linalg.norm(upper - lower) / 2
                confidence = max(0, 1 - distance / max_distance) if max_distance > 0 else 1.0
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = mapping.get('result', 'unknown')
        
        return (best_match, best_confidence)
    
    def detect_all_tubes(self, test_names: List[str]) -> Dict[str, Any]:
        """Capture and detect colors for all tubes"""
        frame = self.capture_frame()
        if frame is None:
            return {"error": "Failed to capture frame", "results": []}
        
        results = self.detect_tube_colors(frame, test_names)
        
        return {
            "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),
            "tube_count": len(results),
            "results": [
                {
                    "tube_id": r.tube_id,
                    "test": r.test_name,
                    "color": r.hex_color,
                    "rgb": list(r.rgb),
                    "result": r.result,
                    "confidence": round(r.confidence, 2)
                }
                for r in results
            ]
        }
    
    def save_debug_image(self, frame: np.ndarray, results: List[ColorResult], path: str):
        """Save debug image with ROIs annotated"""
        debug_frame = frame.copy()
        
        for i, roi in enumerate(self.tube_rois):
            x, y, w, h = roi
            
            # Draw ROI rectangle
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add label
            if i < len(results):
                r = results[i]
                label = f"T{r.tube_id}: {r.result}"
                cv2.putText(debug_frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(path, debug_frame)
        logger.info(f"Saved debug image: {path}")
    
    def shutdown(self):
        """Release camera"""
        if self.camera:
            self.camera.release()
            logger.info("Color detector shutdown")


def test_color_detector():
    """Test function for color detector"""
    config = {
        'tube_rois': [
            [100, 200, 80, 150],
            [200, 200, 80, 150],
            [300, 200, 80, 150],
            [400, 200, 80, 150]
        ],
        'test_mappings': {
            'pH': [
                {'color_range': [[0, 100, 100], [10, 255, 255]], 'result': 'acidic'},
                {'color_range': [[35, 100, 100], [85, 255, 255]], 'result': 'neutral'},
                {'color_range': [[85, 100, 100], [130, 255, 255]], 'result': 'alkaline'}
            ]
        }
    }
    
    detector = ColorDetector(config)
    if detector.initialize(0):
        results = detector.detect_all_tubes(['pH', 'nitrate', 'phosphate', 'potassium'])
        print(f"Detection results: {results}")
        detector.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_color_detector()
