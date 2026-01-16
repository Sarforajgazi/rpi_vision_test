"""
ROI Calibrator for Science Hub
Interactive tool to set up test tube regions (ROIs) for color detection.
Saves ROI positions to use in config.yaml
"""

import cv2
import numpy as np
import yaml
from typing import List, Tuple, Optional

class ROICalibrator:
    """Interactive tool to draw and save test tube ROI regions"""
    
    def __init__(self, camera_device: int = 0):
        self.camera_device = camera_device
        self.camera: Optional[cv2.VideoCapture] = None
        self.rois: List[Tuple[int, int, int, int]] = []  # (x, y, w, h)
        self.drawing = False
        self.start_point = (0, 0)
        self.current_rect = None
        self.frame = None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing rectangles"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_rect = None
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_rect = (
                min(self.start_point[0], x),
                min(self.start_point[1], y),
                abs(x - self.start_point[0]),
                abs(y - self.start_point[1])
            )
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_rect and self.current_rect[2] > 10 and self.current_rect[3] > 10:
                self.rois.append(self.current_rect)
                print(f"✅ Added ROI {len(self.rois)}: {self.current_rect}")
            self.current_rect = None
    
    def run(self) -> List[Tuple[int, int, int, int]]:
        """
        Run interactive ROI calibration.
        
        Controls:
        - Drag mouse: Draw ROI rectangle
        - 'z': Undo last ROI
        - 'c': Clear all ROIs
        - 's': Save ROIs to file
        - 'q': Quit
        
        Returns:
            List of ROI tuples (x, y, width, height)
        """
        self.camera = cv2.VideoCapture(self.camera_device)
        if not self.camera.isOpened():
            print(f"Failed to open camera {self.camera_device}")
            return []
        
        cv2.namedWindow("ROI Calibrator")
        cv2.setMouseCallback("ROI Calibrator", self._mouse_callback)
        
        print("\n" + "="*50)
        print("📦 ROI CALIBRATOR")
        print("="*50)
        print("Draw rectangles around each test tube position.")
        print("Controls:")
        print("  [Drag]  - Draw ROI rectangle")
        print("  [z]     - Undo last ROI")
        print("  [c]     - Clear all ROIs")
        print("  [s]     - Save to config format")
        print("  [q]     - Quit")
        print("="*50 + "\n")
        
        while True:
            ret, self.frame = self.camera.read()
            if not ret:
                break
            
            display = self.frame.copy()
            
            # Draw saved ROIs
            for i, (x, y, w, h) in enumerate(self.rois):
                color = (0, 255, 0)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display, f"Tube {i+1}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show dominant color in ROI
                roi_region = self.frame[y:y+h, x:x+w]
                if roi_region.size > 0:
                    avg_color = roi_region.mean(axis=(0, 1)).astype(int)
                    cv2.rectangle(display, (x, y+h+5), (x+30, y+h+25), 
                                 tuple(avg_color.tolist()), -1)
            
            # Draw current rectangle being drawn
            if self.current_rect:
                x, y, w, h = self.current_rect
                cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # UI
            cv2.rectangle(display, (0, 0), (300, 30), (0, 0, 0), -1)
            cv2.putText(display, f"ROIs: {len(self.rois)} | Draw rectangles", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("ROI Calibrator", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('z') and self.rois:
                removed = self.rois.pop()
                print(f"Removed: {removed}")
            elif key == ord('c'):
                self.rois.clear()
                print("Cleared all ROIs")
            elif key == ord('s'):
                self._save_rois()
            elif key == ord('q'):
                break
        
        self.camera.release()
        cv2.destroyAllWindows()
        
        return self.rois
    
    def _save_rois(self):
        """Save ROIs in config.yaml format"""
        if not self.rois:
            print("No ROIs to save!")
            return
        
        # Save as YAML snippet
        filename = "roi_calibration.yaml"
        config_snippet = {
            'color_detection': {
                'tube_rois': [list(roi) for roi in self.rois]
            }
        }
        
        with open(filename, 'w') as f:
            f.write("# Copy this to your config.yaml\n")
            yaml.dump(config_snippet, f, default_flow_style=False)
        
        print(f"\n✅ Saved {len(self.rois)} ROIs to {filename}")
        print("\nCopy this to config.yaml:")
        print("-" * 40)
        for i, roi in enumerate(self.rois):
            print(f"    - [{roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}]   # Tube {i+1}")
        print("-" * 40)


def main():
    import sys
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    calibrator = ROICalibrator(camera_device=device)
    rois = calibrator.run()
    
    print(f"\nFinal ROIs: {rois}")


if __name__ == "__main__":
    main()
