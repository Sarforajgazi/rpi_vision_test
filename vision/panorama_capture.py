"""
Panorama Capture for Science Hub
Captures and stitches 180-degree panoramic images
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PanoramaResult:
    """Result of panorama capture"""
    success: bool
    image_path: Optional[str]
    frame_count: int
    error: Optional[str] = None


class PanoramaCapture:
    """Captures and stitches panoramic images"""
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.camera_device = config.get('device', 1)
        self.capture_limit = config.get('capture_limit', 8)
        self.interval = config.get('interval', 3)  # seconds between captures
        self.frame_width = config.get('frame_width', 800)
        self.confidence_thresh = config.get('confidence_threshold', 0.6)
        self.output_dir = config.get('output_dir', './data/panoramas')
        
        self.camera: Optional[cv2.VideoCapture] = None
        self.frames: List[np.ndarray] = []
    
    def initialize(self, device: int = None) -> bool:
        """Initialize camera"""
        if device is not None:
            self.camera_device = device
        
        try:
            self.camera = cv2.VideoCapture(self.camera_device)
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {self.camera_device}")
                return False
            
            logger.info(f"Panorama camera {self.camera_device} initialized")
            return True
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            return False
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame maintaining aspect ratio"""
        h, w = frame.shape[:2]
        ratio = self.frame_width / w
        return cv2.resize(frame, (self.frame_width, int(h * ratio)))
    
    def capture_frames_interactive(self, show_preview: bool = True) -> List[np.ndarray]:
        """
        Capture frames interactively with ghost overlay for alignment.
        Press 'q' to quit early.
        """
        if not self.camera or not self.camera.isOpened():
            if not self.initialize():
                return []
        
        self.frames = []
        last_capture = time.time()
        last_frame_captured = None
        
        logger.info(f"Starting panorama capture: {self.capture_limit} frames, {self.interval}s interval")
        
        while len(self.frames) < self.capture_limit:
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            if show_preview:
                display_frame = frame.copy()
                
                # Ghost overlay - shows previous capture for alignment
                if last_frame_captured is not None:
                    display_frame = cv2.addWeighted(frame, 0.6, last_frame_captured, 0.4, 0)
                
                # UI overlay
                cv2.putText(display_frame, f"Captured: {len(self.frames)}/{self.capture_limit}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Rotate slowly. Press 'q' to finish.", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Panorama Capture (Align with Ghost)", display_frame)
            
            current_time = time.time()
            if current_time - last_capture >= self.interval:
                frame_small = self._resize_frame(frame)
                self.frames.append(frame_small)
                last_frame_captured = frame.copy()
                last_capture = current_time
                logger.info(f"Captured frame {len(self.frames)}/{self.capture_limit}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Capture stopped by user")
                break
        
        if show_preview:
            cv2.destroyAllWindows()
        
        return self.frames
    
    def capture_frames_auto(self, num_frames: int = None) -> List[np.ndarray]:
        """
        Capture frames automatically without preview.
        For automated/headless operation.
        """
        if not self.camera or not self.camera.isOpened():
            if not self.initialize():
                return []
        
        num_frames = num_frames or self.capture_limit
        self.frames = []
        
        logger.info(f"Auto-capturing {num_frames} frames...")
        
        for i in range(num_frames):
            # Wait for interval
            time.sleep(self.interval)
            
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            frame_small = self._resize_frame(frame)
            self.frames.append(frame_small)
            logger.info(f"Captured frame {len(self.frames)}/{num_frames}")
        
        return self.frames
    
    def stitch(self, frames: List[np.ndarray] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Stitch captured frames into panorama.
        Returns (success, panorama_image)
        """
        frames = frames or self.frames
        
        if len(frames) < 2:
            logger.error("Need at least 2 frames to stitch")
            return False, None
        
        logger.info(f"Stitching {len(frames)} frames in SCANS mode...")
        
        # Use SCANS mode for linear/horizontal rotation
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        stitcher.setPanoConfidenceThresh(self.confidence_thresh)
        
        status, panorama = stitcher.stitch(frames)
        
        if status != cv2.Stitcher_OK:
            error_msgs = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera params adjustment failed"
            }
            logger.error(f"Stitching failed: {error_msgs.get(status, f'Status {status}')}")
            return False, None
        
        # Clean crop to remove black borders
        cleaned = self._clean_crop(panorama)
        
        logger.info("Stitching successful")
        return True, cleaned
    
    def _clean_crop(self, panorama: np.ndarray) -> np.ndarray:
        """Remove black borders from stitched panorama"""
        # Add border for processing
        stitched = cv2.copyMakeBorder(panorama, 10, 10, 10, 10, 
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Create mask
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        
        # Find largest contour
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            return panorama
        
        c = max(cnts, key=cv2.contourArea)
        
        # Create rectangle mask
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # Iteratively erode to find clean rectangle
        minRect = mask.copy()
        sub = mask.copy()
        
        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)
        
        # Extract final rectangle
        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            return panorama
        
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        
        return stitched[y:y + h, x:x + w]
    
    def capture_and_stitch(self, site_id: int = 0, auto: bool = False) -> PanoramaResult:
        """
        Complete panorama workflow: capture frames and stitch.
        
        Args:
            site_id: Site ID for filename
            auto: If True, capture without preview (headless mode)
        
        Returns:
            PanoramaResult with success status and image path
        """
        # Capture frames
        if auto:
            frames = self.capture_frames_auto()
        else:
            frames = self.capture_frames_interactive()
        
        if len(frames) < 2:
            return PanoramaResult(
                success=False,
                image_path=None,
                frame_count=len(frames),
                error="Not enough frames captured"
            )
        
        # Stitch
        success, panorama = self.stitch(frames)
        
        if not success or panorama is None:
            return PanoramaResult(
                success=False,
                image_path=None,
                frame_count=len(frames),
                error="Stitching failed"
            )
        
        # Save
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/site{site_id}_panorama_{timestamp}.jpg"
        
        cv2.imwrite(filename, panorama)
        logger.info(f"Saved panorama: {filename}")
        
        return PanoramaResult(
            success=True,
            image_path=filename,
            frame_count=len(frames)
        )
    
    def shutdown(self):
        """Release camera"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        logger.info("Panorama capture shutdown")


def test_panorama():
    """Test panorama capture"""
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'device': 1,
        'capture_limit': 8,
        'interval': 3,
        'output_dir': './data/panoramas'
    }
    
    capture = PanoramaCapture(config)
    
    if capture.initialize():
        result = capture.capture_and_stitch(site_id=1, auto=False)
        
        if result.success:
            print(f"✅ Panorama saved: {result.image_path}")
            print(f"   Frames used: {result.frame_count}")
        else:
            print(f"❌ Failed: {result.error}")
        
        capture.shutdown()


if __name__ == "__main__":
    test_panorama()
