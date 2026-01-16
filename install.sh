#!/bin/bash
# Install dependencies for RPi Vision Test

echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install opencv-python numpy pyyaml ultralytics torch torchvision RPi.GPIO

echo ""
echo "Checking available cameras..."
ls -la /dev/video*

echo ""
echo "Installation complete!"
echo "Run: python3 test_vision.py"
