"""
Servo Panorama Capture
Integrates servo rotation with camera capture for automated 180° panoramas
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Import servo controller
try:
    from servo_controller import ServoController
except ImportError:
    from .servo_controller import ServoController

logger = logging.getLogger(__name__)


@dataclass
class ServoPanoramaResult:
    """Result of servo-controlled panorama capture"""
    success: bool
    image_path: Optional[str]
    frame_count: int
    angles: List[float]
    error: Optional[str] = None


class ServoPanorama:
    """
    Captures panorama using servo-controlled camera rotation.
    Servo rotates 180°, camera captures at each position, then stitches.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Camera settings
        self.camera_device = config.get('camera_device', 0)
        self.frame_width = config.get('frame_width', 800)
        self.output_dir = config.get('output_dir', './data/panoramas')
        
        # Servo settings
        self.servo_pin = config.get('servo_pin', 18)
        self.num_positions = config.get('num_positions', 10)
        self.stabilize_time = config.get('stabilize_time', 0.5)  # Time to wait after servo moves
        self.degree_step = config.get('degree_step', 1)  # Degrees per step for smooth movement
        
        # Stitching settings - lower threshold for better success rate
        self.confidence_thresh = config.get('confidence_threshold', 0.3)
        
        # Components
        self.camera: Optional[cv2.VideoCapture] = None
        self.servo: Optional[ServoController] = None
        self.frames: List[np.ndarray] = []
        self.angles: List[float] = []
    
    def initialize(self) -> bool:
        """Initialize camera and servo"""
        success = True
        
        # Initialize camera
        try:
            self.camera = cv2.VideoCapture(self.camera_device)
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {self.camera_device}")
                success = False
            else:
                logger.info(f"Camera {self.camera_device} initialized")
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            success = False
        
        # Initialize servo
        self.servo = ServoController(gpio_pin=self.servo_pin)
        if not self.servo.initialize():
            logger.warning("Servo init failed - will run in simulation mode")
        
        return success
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame maintaining aspect ratio"""
        h, w = frame.shape[:2]
        ratio = self.frame_width / w
        return cv2.resize(frame, (self.frame_width, int(h * ratio)))
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture single frame from camera"""
        if not self.camera or not self.camera.isOpened():
            return None
        
        # Discard a few frames to get fresh image
        for _ in range(3):
            self.camera.read()
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        return self._resize_frame(frame)
    
    def capture_panorama(self, site_id: int = 0) -> ServoPanoramaResult:
        """
        Capture full 180° panorama with servo rotation.
        
        Args:
            site_id: Site ID for filename
        
        Returns:
            ServoPanoramaResult with success status, image path, and angles
        """
        if not self.camera or not self.camera.isOpened():
            return ServoPanoramaResult(
                success=False,
                image_path=None,
                frame_count=0,
                angles=[],
                error="Camera not initialized"
            )
        
        self.frames = []
        self.angles = []
        
        # Calculate step angle
        step = 180.0 / (self.num_positions - 1)
        
        logger.info(f"Starting servo panorama: {self.num_positions} positions")
        
        # Move to start position (0°)
        self.servo.set_angle(0, smooth=True, degree_step=self.degree_step)
        time.sleep(0.5)
        
        # Capture at each position
        for i in range(self.num_positions):
            angle = i * step
            
            # Move servo to position
            self.servo.set_angle(angle, smooth=True, degree_step=self.degree_step)
            time.sleep(self.stabilize_time)  # Wait for camera to stabilize
            
            # Capture frame
            frame = self.capture_frame()
            if frame is not None:
                self.frames.append(frame)
                self.angles.append(angle)
                logger.info(f"Captured {i+1}/{self.num_positions} at {angle:.1f}°")
            else:
                logger.warning(f"Failed to capture at {angle:.1f}°")
        
        # Return servo to start
        self.servo.set_angle(0, smooth=True, degree_step=self.degree_step)
        
        if len(self.frames) < 2:
            return ServoPanoramaResult(
                success=False,
                image_path=None,
                frame_count=len(self.frames),
                angles=self.angles,
                error="Not enough frames captured"
            )
        
        # Stitch frames
        success, panorama = self._stitch(self.frames)
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save individual frames for debugging
        frames_dir = f"{self.output_dir}/frames_{timestamp}"
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(self.frames):
            cv2.imwrite(f"{frames_dir}/frame_{i:02d}_{self.angles[i]:.0f}deg.jpg", frame)
        logger.info(f"Saved {len(self.frames)} individual frames to {frames_dir}")
        
        if not success or panorama is None:
            return ServoPanoramaResult(
                success=False,
                image_path=None,
                frame_count=len(self.frames),
                angles=self.angles,
                error=f"Stitching failed - check individual frames in {frames_dir}"
            )
        
        # Save panorama
        filename = f"{self.output_dir}/site{site_id}_servo_panorama_{timestamp}.jpg"
        
        cv2.imwrite(filename, panorama)
        logger.info(f"Saved panorama: {filename}")
        
        return ServoPanoramaResult(
            success=True,
            image_path=filename,
            frame_count=len(self.frames),
            angles=self.angles
        )
    
    def _stitch(self, frames: List[np.ndarray]) -> Tuple[bool, Optional[np.ndarray]]:
        """Stitch frames into panorama"""
        logger.info(f"Stitching {len(frames)} frames...")
        
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        stitcher.setPanoConfidenceThresh(self.confidence_thresh)
        
        status, panorama = stitcher.stitch(frames)
        
        if status != cv2.Stitcher_OK:
            logger.error(f"Stitching failed with status {status}")
            return False, None
        
        # Clean crop
        panorama = self._clean_crop(panorama)
        
        return True, panorama
    
    def _clean_crop(self, panorama: np.ndarray) -> np.ndarray:
        """Remove black borders from stitched panorama"""
        stitched = cv2.copyMakeBorder(panorama, 10, 10, 10, 10,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            return panorama
        
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        minRect = mask.copy()
        sub = mask.copy()
        
        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)
        
        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            return panorama
        
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        
        return stitched[y:y + h, x:x + w]
    
    def shutdown(self):
        """Release resources"""
        if self.camera:
            self.camera.release()
        if self.servo:
            self.servo.shutdown()
        logger.info("Servo panorama shutdown")


def main():
    """Test servo panorama capture"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Servo-controlled panorama capture")
    parser.add_argument('--camera', type=int, default=0, help='Camera device number')
    parser.add_argument('--servo-pin', type=int, default=18, help='GPIO pin for servo')
    parser.add_argument('--positions', type=int, default=8, help='Number of capture positions')
    parser.add_argument('--site', type=int, default=0, help='Site ID')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = {
        'camera_device': args.camera,
        'servo_pin': args.servo_pin,
        'num_positions': args.positions,
        'output_dir': './data/panoramas'
    }
    
    print("\n" + "="*50)
    print("🔄 SERVO PANORAMA CAPTURE")
    print("="*50)
    print(f"Camera: {args.camera}")
    print(f"Servo GPIO: {args.servo_pin}")
    print(f"Positions: {args.positions}")
    print("="*50 + "\n")
    
    pano = ServoPanorama(config)
    
    if pano.initialize():
        result = pano.capture_panorama(site_id=args.site)
        
        if result.success:
            print(f"\n✅ Panorama saved: {result.image_path}")
            print(f"   Frames: {result.frame_count}")
            print(f"   Angles: {[f'{a:.1f}°' for a in result.angles]}")
        else:
            print(f"\n❌ Failed: {result.error}")
        
        pano.shutdown()
    else:
        print("Failed to initialize")


if __name__ == "__main__":
    main()
