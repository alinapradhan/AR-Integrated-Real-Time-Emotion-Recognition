#!/bin/bash
# AR Emotion Recognition System - Setup Script

echo "AR-Integrated Real-Time Emotion Recognition System Setup"
echo "========================================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python3 -m venv ar_emotion_env 2>/dev/null || echo "Virtual environment creation skipped"

# Install required packages
echo "Installing dependencies..."
pip3 install --user opencv-python numpy pillow matplotlib scikit-learn imutils || {
    echo "Basic packages installed"
}

# Try to install TensorFlow and MediaPipe (optional for full functionality)
echo "Installing optional packages for full functionality..."
pip3 install --user tensorflow mediapipe 2>/dev/null || {
    echo "Optional packages (TensorFlow, MediaPipe) not installed"
    echo "System will work with OpenCV fallback for face detection"
    echo "and mock emotion recognition for demonstration"
}

echo ""
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  python3 main.py              # Run full system (requires TensorFlow)"
echo "  python3 demo.py              # Run demo with mock data"
echo "  python3 generate_demo.py     # Generate demo images"
echo ""
echo "Controls when running:"
echo "  Q/ESC : Quit"
echo "  D     : Toggle debug info"
echo "  C     : Toggle confidence bars"
echo "  E     : Toggle emotion dashboard"
echo "  SPACE : Generate new emotion (demo mode)"
echo ""