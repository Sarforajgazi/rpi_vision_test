# RPi Vision Test

Minimal folder for testing all 3 cameras on Raspberry Pi.

## Folder Structure

```
rpi_vision_test/
├── config.yaml          ← Camera & detection settings
├── test_vision.py       ← Main test script
├── install.sh           ← Install dependencies
├── README.md
├── vision/
│   ├── color_detector.py
│   ├── sample_classifier.py
│   └── panorama_capture.py
├── models/
│   ├── soil_classifier/
│   └── rock_classifier_micro/
└── data/                ← Output folder
```

## Setup on RPi

```bash
# 1. Copy this folder to RPi
scp -r rpi_vision_test pi@<RPI_IP>:/home/pi/

# 2. SSH into RPi
ssh pi@<RPI_IP>

# 3. Install dependencies
cd /home/pi/rpi_vision_test
chmod +x install.sh
./install.sh

# 4. Check camera devices
ls /dev/video*

# 5. Update config.yaml with correct camera device numbers
nano config.yaml
```

## Run Tests

```bash
# Test all 3 cameras (sequential)
python3 test_vision.py

# Test all 3 cameras (parallel - requires 3 separate cameras)
python3 test_vision.py --parallel
```

## Output

Results saved to `data/vision_test/`:
- `vision_test_*.json` - Combined JSON results
- `microscopic_*.jpg` - Sample images
- `site0_panorama_*.jpg` - Panorama images

## Camera Configuration

Edit `config.yaml`:
```yaml
cameras:
  endoscopic:  { device: 0 }  # Test tube color detection
  microscopic: { device: 1 }  # Soil/rock ML classification
  panorama:    { device: 2 }  # 180° panorama
```
