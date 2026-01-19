"""
Direct Camera Stream Display
Run directly on Jetson to view camera output on local display
"""

import cv2
import yaml
import argparse
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load camera config"""
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except:
        return {
            'cameras': {
                'endoscopic': {'device': 2},
                'microscopic': {'device': 4},
                'panorama': {'device': 0}
            }
        }


def open_camera(cam_id: int) -> Optional[cv2.VideoCapture]:
    """Open camera by ID"""
    cap = cv2.VideoCapture(cam_id)
    if cap.isOpened():
        logger.info(f"Camera {cam_id} opened successfully")
        return cap
    else:
        logger.error(f"Failed to open camera {cam_id}")
        return None


def stream_single_camera(cam_id: int):
    """Stream a single camera to display"""
    cap = open_camera(cam_id)
    if not cap:
        print(f"Error: Could not open camera {cam_id}")
        return
    
    window_name = f"Camera {cam_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    print(f"\n{'='*50}")
    print(f"📷 Streaming Camera {cam_id}")
    print(f"Press 'q' to quit, 's' to save snapshot")
    print(f"{'='*50}\n")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read from camera {cam_id}")
            break
        
        # Add camera info overlay
        cv2.putText(frame, f"Camera {cam_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit | 's' to snapshot", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"snapshot_cam{cam_id}_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def stream_all_cameras():
    """Stream all available cameras in a grid"""
    config = load_config()
    cam_config = config.get('cameras', {})
    
    # Get camera device IDs from config
    cam_ids = []
    cam_names = {}
    for name, cfg in cam_config.items():
        device_id = cfg.get('device', 0)
        cam_ids.append(device_id)
        cam_names[device_id] = name
    
    if not cam_ids:
        cam_ids = [0, 2, 4]  # Default cameras
    
    # Open all cameras
    cameras: Dict[int, cv2.VideoCapture] = {}
    for cam_id in cam_ids:
        cap = open_camera(cam_id)
        if cap:
            cameras[cam_id] = cap
    
    if not cameras:
        print("Error: No cameras available!")
        return
    
    cv2.namedWindow("All Cameras", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("All Cameras", 1200, 800)
    
    print(f"\n{'='*50}")
    print(f"📷 Streaming {len(cameras)} cameras")
    print(f"Cameras: {list(cameras.keys())}")
    print(f"Press 'q' to quit, 's' to save snapshots")
    print(f"{'='*50}\n")
    
    import numpy as np
    
    while True:
        frames = []
        for cam_id, cap in cameras.items():
            ret, frame = cap.read()
            if ret:
                # Resize for grid display
                frame = cv2.resize(frame, (400, 300))
                # Add label
                name = cam_names.get(cam_id, f"Camera {cam_id}")
                cv2.putText(frame, name, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                frames.append(frame)
            else:
                # Placeholder for failed read
                placeholder = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Camera {cam_id} - No Signal", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frames.append(placeholder)
        
        if frames:
            # Arrange side by side (3 columns - single row)
            import math
            cols = 3
            rows = math.ceil(len(frames) / cols)
            
            # Pad frames if needed
            while len(frames) < rows * cols:
                frames.append(np.zeros((300, 400, 3), dtype=np.uint8))
            
            # Create grid
            grid_rows = []
            for i in range(rows):
                row = np.hstack(frames[i*cols:(i+1)*cols])
                grid_rows.append(row)
            grid = np.vstack(grid_rows)
            
            cv2.imshow("All Cameras", grid)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            for cam_id, cap in cameras.items():
                ret, frame = cap.read()
                if ret:
                    filename = f"snapshot_{cam_names.get(cam_id, cam_id)}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved: {filename}")
    
    # Cleanup
    for cap in cameras.values():
        cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Direct Camera Stream Display')
    parser.add_argument('-c', '--camera', type=int, default=None,
                       help='Camera device ID (e.g., 0, 2, 4). If not specified, shows all cameras.')
    parser.add_argument('-a', '--all', action='store_true',
                       help='Show all cameras in grid view')
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("🎥 JETSON DIRECT CAMERA VIEWER")
    print("="*50)
    
    if args.camera is not None:
        # Single camera mode
        stream_single_camera(args.camera)
    else:
        # All cameras mode
        stream_all_cameras()


if __name__ == '__main__':
    main()
