"""
Sample Classifier for Science Hub
Classifies soil/rock samples using microscopic camera and ML models
Supports multiple classifiers: soil, rock_macro, rock_micro
Supports YOLO, ResNet, ONNX, TFLite model formats
"""

import cv2
import numpy as np
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLOClassifier:
    """YOLO-based classifier (using ultralytics)"""
    
    def __init__(self, model_path: str, classes: List[str], confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False
    
    def load(self) -> bool:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            # Try .pt file
            pt_file = os.path.join(self.model_path, "model.pt")
            if os.path.exists(pt_file):
                self.model = YOLO(pt_file)
                self.is_loaded = True
                # Update classes from model if available
                if hasattr(self.model, 'names') and self.model.names:
                    self.classes = list(self.model.names.values())
                logger.info(f"Loaded YOLO model from {pt_file}")
                logger.info(f"Classes: {self.classes}")
                return True
            
            logger.warning(f"Model not found at {pt_file}")
            return False
            
        except ImportError:
            logger.warning("ultralytics not installed, YOLO models unavailable")
            return False
        except Exception as e:
            logger.error(f"YOLO load error: {e}")
            return False
    
    def classify(self, frame: np.ndarray) -> Dict[str, Any]:
        """Classify using YOLO"""
        if not self.is_loaded or self.model is None:
            return self._demo_classify(frame)
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                
                # Classification model (has probs)
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    top_class = int(probs.top1)
                    confidence = float(probs.top1conf)
                    
                    class_name = self.model.names.get(top_class, "unknown")
                    
                    return {
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "all_predictions": {
                            self.model.names[i]: round(float(probs.data[i]), 3)
                            for i in range(len(probs.data))
                        },
                        "is_confident": confidence >= self.confidence_threshold
                    }
                
                # Detection model (has boxes)
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        # Get highest confidence detection
                        best_idx = boxes.conf.argmax()
                        class_id = int(boxes.cls[best_idx])
                        confidence = float(boxes.conf[best_idx])
                        class_name = self.model.names.get(class_id, "unknown")
                        
                        # Count all detections
                        detections = {}
                        for i in range(len(boxes)):
                            cls = self.model.names.get(int(boxes.cls[i]), "unknown")
                            if cls not in detections:
                                detections[cls] = 0
                            detections[cls] += 1
                        
                        return {
                            "class": class_name,
                            "confidence": round(confidence, 3),
                            "detections": detections,
                            "detection_count": len(boxes),
                            "is_confident": confidence >= self.confidence_threshold
                        }
            
            return {"class": "none_detected", "confidence": 0.0, "is_confident": False}
            
        except Exception as e:
            logger.error(f"YOLO classify error: {e}")
            return {"error": str(e), "class": "unknown", "confidence": 0.0}
    
    def _demo_classify(self, frame: np.ndarray) -> Dict[str, Any]:
        """Demo classification fallback"""
        if len(self.classes) > 0:
            idx = hash(frame.tobytes()) % len(self.classes)
            return {
                "class": self.classes[idx],
                "confidence": 0.5,
                "demo_mode": True
            }
        return {"class": "unknown", "confidence": 0.0, "demo_mode": True}


class ResNetClassifier:
    """ResNet/PyTorch classifier"""
    
    def __init__(self, model_path: str, classes: List[str], confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False
    
    def load(self) -> bool:
        """Load PyTorch model"""
        try:
            import torch
            
            pt_file = os.path.join(self.model_path, "model.pt")
            if os.path.exists(pt_file):
                self.model = torch.load(pt_file, map_location='cpu')
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                self.is_loaded = True
                logger.info(f"Loaded ResNet model from {pt_file}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"ResNet load error: {e}")
            return False
    
    def classify(self, frame: np.ndarray) -> Dict[str, Any]:
        """Classify using ResNet"""
        if not self.is_loaded:
            return self._demo_classify(frame)
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Preprocess
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
            # To tensor
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
            
            with torch.no_grad():
                output = self.model(tensor)
                probs = F.softmax(output, dim=1)[0]
                
                class_idx = probs.argmax().item()
                confidence = probs[class_idx].item()
                
                return {
                    "class": self.classes[class_idx] if class_idx < len(self.classes) else "unknown",
                    "confidence": round(confidence, 3),
                    "all_predictions": {
                        self.classes[i]: round(probs[i].item(), 3)
                        for i in range(min(len(probs), len(self.classes)))
                    },
                    "is_confident": confidence >= self.confidence_threshold
                }
                
        except Exception as e:
            logger.error(f"ResNet classify error: {e}")
            return {"error": str(e), "class": "unknown", "confidence": 0.0}
    
    def _demo_classify(self, frame: np.ndarray) -> Dict[str, Any]:
        """Demo classification fallback"""
        if len(self.classes) > 0:
            idx = hash(frame.tobytes()) % len(self.classes)
            return {
                "class": self.classes[idx],
                "confidence": 0.5,
                "demo_mode": True
            }
        return {"class": "unknown", "confidence": 0.0, "demo_mode": True}


class SingleClassifier:
    """Wrapper for single classifier of any type"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model_path = config.get('model_path', '')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.classes = config.get('classes', [])
        self.model_type = config.get('model_type', 'yolo')
        self.classifier = None
        self.is_loaded = False
    
    def load(self) -> bool:
        """Load the appropriate classifier"""
        if self.model_type == 'yolo':
            self.classifier = YOLOClassifier(
                self.model_path, self.classes, self.confidence_threshold
            )
        elif self.model_type == 'resnet':
            self.classifier = ResNetClassifier(
                self.model_path, self.classes, self.confidence_threshold
            )
        else:
            logger.warning(f"Unknown model type: {self.model_type}, defaulting to YOLO")
            self.classifier = YOLOClassifier(
                self.model_path, self.classes, self.confidence_threshold
            )
        
        self.is_loaded = self.classifier.load()
        if not self.is_loaded:
            logger.warning(f"{self.name}: Running in demo mode")
        return self.is_loaded
    
    def classify(self, frame: np.ndarray) -> Dict[str, Any]:
        """Classify the frame"""
        if self.classifier:
            result = self.classifier.classify(frame)
            result["classifier"] = self.name
            return result
        return {"classifier": self.name, "class": "unknown", "confidence": 0.0, "error": "No classifier"}


class SampleClassifier:
    """
    Multi-model classifier for Science Hub.
    Supports: soil, rock_macro, rock_micro classifiers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.classifiers: Dict[str, SingleClassifier] = {}
        self.camera = None
        self.camera_device = 1
        
        # Initialize classifiers from config
        classifiers_config = config.get('classifiers', {})
        
        # Support both old and new config formats
        if not classifiers_config and 'model_path' in config:
            classifiers_config = {'soil': config}
        
        for name, classifier_config in classifiers_config.items():
            self.classifiers[name] = SingleClassifier(name, classifier_config)
    
    def load_model(self) -> bool:
        """Load all classifier models"""
        success = True
        for name, classifier in self.classifiers.items():
            if not classifier.load():
                success = False
        return success
    
    def initialize_camera(self, device: int = 1) -> bool:
        """Initialize microscopic camera"""
        self.camera_device = device
        try:
            self.camera = cv2.VideoCapture(device)
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {device}")
                return False
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            logger.info(f"Camera {device} initialized")
            return True
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from camera"""
        if not self.camera or not self.camera.isOpened():
            if not self.initialize_camera(self.camera_device):
                return None
        
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None
        return frame
    
    def classify(self, frame: np.ndarray, classifier_name: str = "soil") -> Dict[str, Any]:
        """Classify using specified classifier"""
        if classifier_name not in self.classifiers:
            return {"error": f"Unknown classifier: {classifier_name}"}
        
        return self.classifiers[classifier_name].classify(frame)
    
    def classify_all(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run all classifiers on the frame"""
        results = {}
        for name, classifier in self.classifiers.items():
            results[name] = classifier.classify(frame)
        return results
    
    def classify_and_capture(self, classifier_name: str = "soil") -> Dict[str, Any]:
        """Capture frame and classify"""
        frame = self.capture_frame()
        if frame is None:
            return {"error": "Failed to capture frame", "class": "unknown"}
        
        result = self.classify(frame, classifier_name)
        result["frame_captured"] = True
        return result
    
    def classify_sample(self) -> Dict[str, Any]:
        """
        Run full sample classification (soil + microscopic rocks).
        Returns combined results from both classifiers.
        """
        frame = self.capture_frame()
        if frame is None:
            return {"error": "Failed to capture frame"}
        
        results = {
            "sample_type": "unknown",
            "sample_confidence": 0.0,
            "rock_type": "unknown", 
            "rock_confidence": 0.0
        }
        
        # Classify soil
        if "soil" in self.classifiers:
            soil_result = self.classifiers["soil"].classify(frame)
            results["sample_type"] = soil_result.get("class", "unknown")
            results["sample_confidence"] = soil_result.get("confidence", 0.0)
            results["soil_details"] = soil_result
        
        # Classify microscopic rock
        if "rock_micro" in self.classifiers:
            rock_result = self.classifiers["rock_micro"].classify(frame)
            results["rock_type"] = rock_result.get("class", "unknown")
            results["rock_confidence"] = rock_result.get("confidence", 0.0)
            results["rock_details"] = rock_result
        
        return results
    
    def save_sample_image(self, frame: np.ndarray, site_id: int, output_dir: str) -> str:
        """Save sample image to disk"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/site{site_id}_micro_{timestamp}.jpg"
        
        cv2.imwrite(filename, frame)
        logger.info(f"Saved sample image: {filename}")
        
        return filename
    
    def shutdown(self):
        """Release resources"""
        if self.camera:
            self.camera.release()
        logger.info("Sample classifier shutdown")


def test_classifier():
    """Test the sample classifier"""
    config = {
        'classifiers': {
            'soil': {
                'model_path': './models/soil_classifier',
                'confidence_threshold': 0.5,
                'model_type': 'yolo',
                'classes': ['Alluvial Soil', 'Black Soil', 'Cinder Soil', 'Clay Soil', 
                           'Laterite Soil', 'Peat Soil', 'Red Soil', 'Yellow Soil']
            },
            'rock_micro': {
                'model_path': './models/rock_classifier_micro',
                'confidence_threshold': 0.3,
                'model_type': 'yolo',
                'classes': ['Feldspar', 'Kuarsa', 'Litik']
            }
        }
    }
    
    classifier = SampleClassifier(config)
    classifier.load_model()
    
    print("Available classifiers:", list(classifier.classifiers.keys()))
    
    # Test with dummy frame if no camera
    print("\nTesting with dummy frame...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    for name in classifier.classifiers:
        result = classifier.classify(dummy_frame, name)
        print(f"\n{name}: {result}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_classifier()
