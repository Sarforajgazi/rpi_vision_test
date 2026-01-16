"""
Vision Test - Run all 3 cameras together with Servo Panorama
Outputs combined JSON data from:
1. Color Detection (endoscopic camera) - Device 2
2. Sample Classifier (microscopic camera) - Device 4
3. Servo Panorama Capture (panorama camera) - Device 0
"""

import cv2
import json
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import yaml

# Import vision modules
from vision.color_detector import ColorDetector
from vision.sample_classifier import SampleClassifier
from servo_panorama import ServoPanorama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisionTest:
    """Run all 3 cameras together and output JSON"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Output directory
        self.output_dir = "./data/vision_test"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def run_color_detection(self) -> Dict[str, Any]:
        """Run color detection on test tubes"""
        logger.info("🎨 Starting Color Detection...")
        
        try:
            config = self.config.get('color_detection', {})
            detector = ColorDetector(config)
            
            camera_device = self.config.get('cameras', {}).get('endoscopic', {}).get('device', 2)
            if not detector.initialize(camera_device):
                return {"error": "Camera init failed", "camera": camera_device}
            
            test_names = ['xanthophyll', 'ethidium_bromide', 'labile_carbon', 'buret_test']
            result = detector.detect_all_tubes(test_names)
            
            detector.shutdown()
            logger.info(f"✅ Color Detection complete: {len(result.get('results', []))} tubes")
            return result
            
        except Exception as e:
            logger.error(f"Color detection error: {e}")
            return {"error": str(e)}
    
    def run_sample_classifier(self) -> Dict[str, Any]:
        """Run soil/rock classification"""
        logger.info("🔬 Starting Sample Classification...")
        
        try:
            classifier_config = self.config.get('classifiers', {})
            classifier = SampleClassifier({'classifiers': classifier_config})
            classifier.load_model()
            
            camera_device = self.config.get('cameras', {}).get('microscopic', {}).get('device', 4)
            if not classifier.initialize_camera(camera_device):
                return {"error": "Camera init failed", "camera": camera_device}
            
            # Capture and classify
            frame = classifier.capture_frame()
            if frame is None:
                return {"error": "Failed to capture frame"}
            
            # Run all classifiers
            results = classifier.classify_all(frame)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"{self.output_dir}/microscopic_{timestamp}.jpg"
            cv2.imwrite(img_path, frame)
            results["image_path"] = img_path
            
            classifier.shutdown()
            logger.info(f"✅ Sample Classification complete")
            return results
            
        except Exception as e:
            logger.error(f"Sample classifier error: {e}")
            return {"error": str(e)}
    
    def run_panorama(self) -> Dict[str, Any]:
        """Capture panorama with servo rotation"""
        logger.info("📷 Starting Servo Panorama Capture...")
        
        try:
            # Get config values
            servo_config = self.config.get('servo', {})
            camera_config = self.config.get('cameras', {}).get('panorama', {})
            
            panorama_config = {
                'camera_device': camera_config.get('device', 0),
                'servo_pin': servo_config.get('gpio_pin', 18),
                'num_positions': servo_config.get('num_positions', 12),
                'stabilize_time': servo_config.get('stabilize_time', 1.0),
                'output_dir': self.output_dir
            }
            
            pano = ServoPanorama(panorama_config)
            if not pano.initialize():
                return {"error": "Camera/Servo init failed", "camera": panorama_config['camera_device']}
            
            # Capture with servo rotation
            result = pano.capture_panorama(site_id=0)
            pano.shutdown()
            
            if result.success:
                logger.info(f"✅ Servo Panorama complete: {result.image_path}")
                return {
                    "success": True,
                    "image_path": result.image_path,
                    "frames_captured": result.frame_count,
                    "angles": result.angles
                }
            else:
                return {"success": False, "error": result.error}
                
        except Exception as e:
            logger.error(f"Panorama error: {e}")
            return {"error": str(e)}
    
    def run_all_sequential(self) -> Dict[str, Any]:
        """Run all cameras one by one (safer, uses one camera at a time)"""
        timestamp = datetime.now().isoformat()
        
        results = {
            "timestamp": timestamp,
            "mode": "sequential",
            "color_detection": self.run_color_detection(),
            "sample_classification": self.run_sample_classifier(),
            "panorama": self.run_panorama()
        }
        
        return results
    
    def run_all_parallel(self) -> Dict[str, Any]:
        """Run all cameras in parallel (faster, requires 3 separate cameras)"""
        timestamp = datetime.now().isoformat()
        results = {
            "timestamp": timestamp,
            "mode": "parallel"
        }
        
        threads = []
        
        def run_and_store(name, func):
            result = func()
            with self.lock:
                results[name] = result
        
        # Start threads
        t1 = threading.Thread(target=run_and_store, args=("color_detection", self.run_color_detection))
        t2 = threading.Thread(target=run_and_store, args=("sample_classification", self.run_sample_classifier))
        t3 = threading.Thread(target=run_and_store, args=("panorama", self.run_panorama))
        
        threads = [t1, t2, t3]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/vision_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Results saved to: {filename}")
        return filename
    
    def print_results(self, results: Dict[str, Any]):
        """Pretty print results to console"""
        print("\n" + "="*60)
        print("VISION TEST RESULTS")
        print("="*60)
        print(json.dumps(results, indent=2, default=str))
        print("="*60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test all 3 cameras together")
    parser.add_argument('--parallel', action='store_true', help='Run cameras in parallel')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 VISION TEST - All 3 Cameras + Servo")
    print("="*60)
    print("Cameras: Color=2, Microscopic=4, Panorama=0")
    print("Servo: GPIO 18")
    print("="*60)
    
    tester = VisionTest(args.config)
    
    if args.parallel:
        print("Mode: PARALLEL (requires 3 separate cameras)")
        results = tester.run_all_parallel()
    else:
        print("Mode: SEQUENTIAL (reuses cameras if needed)")
        results = tester.run_all_sequential()
    
    # Save and print
    output_file = tester.save_results(results)
    tester.print_results(results)
    
    print(f"\n✅ Test complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()
