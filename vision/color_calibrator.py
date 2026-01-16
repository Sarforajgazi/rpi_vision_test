"""
Color Calibrator for Science Hub
Interactive tool to pick colors and find HSV ranges for test tube detection.
Use this to calibrate the color_detector.py settings.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class ColorCalibrator:
    """Interactive color picker for calibrating test tube color ranges"""
    
    def __init__(self, camera_device: int = 0):
        self.camera_device = camera_device
        self.camera: Optional[cv2.VideoCapture] = None
        self.clicked = False
        self.bgr_color = (0, 0, 0)
        self.hsv_color = (0, 0, 0)
        self.position = (0, 0)
        self.color_history: List[dict] = []
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param.get("frame")
            if frame is not None:
                self.bgr_color = tuple(frame[y, x].tolist())  # (B, G, R)
                self.position = (x, y)
                self.clicked = True
                
                # Calculate HSV
                hsv = cv2.cvtColor(np.uint8([[self.bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
                self.hsv_color = tuple(hsv.tolist())
                
                # Store in history
                self.color_history.append({
                    'position': (x, y),
                    'bgr': self.bgr_color,
                    'hsv': self.hsv_color,
                    'hex': f"#{self.bgr_color[2]:02X}{self.bgr_color[1]:02X}{self.bgr_color[0]:02X}"
                })
                
                logger.info(f"Picked color at ({x}, {y}): BGR={self.bgr_color}, HSV={self.hsv_color}")
    
    def run(self) -> List[dict]:
        """
        Run interactive color calibration.
        
        Controls:
        - Click: Pick color at that point
        - 'c': Clear last pick
        - 's': Save color history to file
        - 'q': Quit
        
        Returns:
            List of picked colors with BGR, HSV values
        """
        self.camera = cv2.VideoCapture(self.camera_device)
        if not self.camera.isOpened():
            logger.error(f"Failed to open camera {self.camera_device}")
            return []
        
        cv2.namedWindow("Color Calibrator")
        param = {"frame": None}
        cv2.setMouseCallback("Color Calibrator", self._mouse_callback, param)
        
        print("\n" + "="*50)
        print("🎨 COLOR CALIBRATOR")
        print("="*50)
        print("Click on test tubes to get their color values.")
        print("Controls:")
        print("  [Click] - Pick color at point")
        print("  [c]     - Clear last pick")
        print("  [s]     - Save history to file")
        print("  [q]     - Quit")
        print("="*50 + "\n")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            param["frame"] = frame
            display = frame.copy()
            
            if self.clicked:
                x, y = self.position
                B, G, R = self.bgr_color
                H, S, V = self.hsv_color
                
                # Draw circle at clicked position
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                
                # Draw info box
                cv2.rectangle(display, (0, 0), (400, 80), (0, 0, 0), -1)
                cv2.putText(display, f"BGR: ({B}, {G}, {R})", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, f"HSV: ({H}, {S}, {V})", (10, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, f"Picked: {len(self.color_history)} colors", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show color preview box
                cv2.rectangle(display, (350, 5), (395, 50), self.bgr_color, -1)
                cv2.rectangle(display, (350, 5), (395, 50), (255, 255, 255), 2)
            
            # Draw history dots
            for i, color in enumerate(self.color_history[-10:]):  # Show last 10
                pos = color['position']
                cv2.circle(display, pos, 3, (0, 0, 255), -1)
            
            cv2.imshow("Color Calibrator", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if self.color_history:
                    removed = self.color_history.pop()
                    print(f"Removed: {removed}")
                self.clicked = bool(self.color_history)
                if self.color_history:
                    last = self.color_history[-1]
                    self.bgr_color = last['bgr']
                    self.hsv_color = last['hsv']
                    self.position = last['position']
            elif key == ord('s'):
                self._save_history()
            elif key == ord('q'):
                break
        
        self.camera.release()
        cv2.destroyAllWindows()
        
        return self.color_history
    
    def _save_history(self):
        """Save color history to file"""
        if not self.color_history:
            print("No colors to save!")
            return
        
        filename = "color_calibration.txt"
        with open(filename, 'w') as f:
            f.write("# Color Calibration Results\n")
            f.write("# Use these HSV values in config.yaml color_range settings\n\n")
            
            for i, color in enumerate(self.color_history, 1):
                f.write(f"Color {i}:\n")
                f.write(f"  Position: {color['position']}\n")
                f.write(f"  BGR: {color['bgr']}\n")
                f.write(f"  HSV: {color['hsv']}  # Use this for color_range\n")
                f.write(f"  HEX: {color['hex']}\n")
                f.write(f"  Suggested range: [[{color['hsv'][0]-10}, {max(0,color['hsv'][1]-50)}, {max(0,color['hsv'][2]-50)}], [{color['hsv'][0]+10}, 255, 255]]\n")
                f.write("\n")
        
        print(f"✅ Saved {len(self.color_history)} colors to {filename}")
    
    def get_suggested_range(self, hsv: Tuple[int, int, int], tolerance: int = 15) -> Tuple[list, list]:
        """Get suggested HSV range for a color"""
        h, s, v = hsv
        lower = [max(0, h - tolerance), max(0, s - 50), max(0, v - 50)]
        upper = [min(179, h + tolerance), 255, 255]
        return lower, upper


def test_calibrator():
    """Run the color calibrator"""
    import sys
    
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    calibrator = ColorCalibrator(camera_device=device)
    colors = calibrator.run()
    
    print("\n" + "="*50)
    print("CAPTURED COLORS:")
    print("="*50)
    for i, c in enumerate(colors, 1):
        lower, upper = calibrator.get_suggested_range(c['hsv'])
        print(f"{i}. HSV: {c['hsv']} → Range: [{lower}, {upper}]")
    print("="*50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_calibrator()
