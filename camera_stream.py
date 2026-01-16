"""
Camera Stream Server
MJPEG streaming for all 3 cameras - viewable in browser
"""

from flask import Flask, Response, render_template_string
import cv2
import threading
import logging
import yaml
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Camera streams
cameras: Dict[int, cv2.VideoCapture] = {}
locks: Dict[int, threading.Lock] = {}


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


def get_camera(cam_id: int) -> Optional[cv2.VideoCapture]:
    """Get or create camera capture"""
    if cam_id not in cameras:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            cameras[cam_id] = cap
            locks[cam_id] = threading.Lock()
            logger.info(f"Camera {cam_id} opened")
        else:
            logger.error(f"Failed to open camera {cam_id}")
            return None
    return cameras[cam_id]


def generate_frames(cam_id: int):
    """Generate MJPEG frames for streaming"""
    cap = get_camera(cam_id)
    if not cap:
        return
    
    while True:
        with locks[cam_id]:
            ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read from camera {cam_id}")
            break
        
        # Resize for faster streaming
        frame = cv2.resize(frame, (640, 480))
        
        # Add camera label
        cv2.putText(frame, f"Camera {cam_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# HTML template for viewing all cameras
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RPi Camera Streams</title>
    <style>
        body { font-family: Arial; background: #1a1a2e; color: white; margin: 20px; }
        h1 { text-align: center; color: #00ff88; }
        .cameras { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
        .camera-box { background: #16213e; border-radius: 10px; padding: 15px; }
        .camera-box h3 { margin: 0 0 10px 0; color: #00ff88; }
        img { border-radius: 5px; max-width: 100%; }
        .status { margin-top: 20px; text-align: center; color: #888; }
    </style>
</head>
<body>
    <h1>🎥 RPi Camera Streams</h1>
    <div class="cameras">
        <div class="camera-box">
            <h3>📷 Camera 0 - Panorama</h3>
            <img src="/stream/0" width="400">
        </div>
        <div class="camera-box">
            <h3>🧪 Camera 2 - Color Detection</h3>
            <img src="/stream/2" width="400">
        </div>
        <div class="camera-box">
            <h3>🔬 Camera 4 - Microscopic</h3>
            <img src="/stream/4" width="400">
        </div>
    </div>
    <p class="status">Streaming from Raspberry Pi</p>
</body>
</html>
"""


@app.route('/')
def index():
    """Main page with all camera streams"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/stream/<int:cam_id>')
def video_stream(cam_id):
    """Stream video from specific camera"""
    return Response(generate_frames(cam_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot/<int:cam_id>')
def snapshot(cam_id):
    """Get single frame from camera"""
    cap = get_camera(cam_id)
    if not cap:
        return "Camera not available", 404
    
    with locks[cam_id]:
        ret, frame = cap.read()
    
    if not ret:
        return "Failed to capture", 500
    
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


def cleanup():
    """Release all cameras"""
    for cam_id, cap in cameras.items():
        cap.release()
        logger.info(f"Camera {cam_id} released")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera Stream Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("📡 CAMERA STREAM SERVER")
    print("="*50)
    print(f"Running on http://0.0.0.0:{args.port}")
    print(f"View streams at: http://<RPI_IP>:{args.port}")
    print("="*50 + "\n")
    
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        cleanup()