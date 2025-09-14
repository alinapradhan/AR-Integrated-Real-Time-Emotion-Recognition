# AR-Integrated Real-Time Emotion Recognition System

This project implements a real-time facial emotion recognition system that utilizes a Convolutional Neural Network (CNN) to detect emotions from live camera feeds and integrates with Augmented Reality (AR) technology to display corresponding emoji overlays above each detected face.

## Features

- **Real-time emotion detection** from live camera feeds
- **AR emoji overlays** displayed above detected faces
- **Multiple emotion recognition**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Performance optimization** for mobile devices and AR glasses
- **Robust face detection** using MediaPipe and OpenCV
- **Smooth prediction filtering** to reduce jitter
- **Interactive controls** for customizing display
- **Session analytics** and performance monitoring

## Requirements

### System Requirements
- Python 3.7+
- Camera/webcam access
- Minimum 4GB RAM (8GB recommended)
- GPU acceleration (optional but recommended)

### Dependencies
```
opencv-python==4.8.1.78
tensorflow==2.13.0
numpy==1.24.3
mediapipe==0.10.7
pillow==10.0.1
matplotlib==3.7.2
scikit-learn==1.3.0
opencv-contrib-python==4.8.1.78
imutils==0.5.4
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alinapradhan/AR-Integrated-Real-Time-Emotion-Recognition.git
cd AR-Integrated-Real-Time-Emotion-Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

### Basic Usage
```bash
python main.py
```

### Advanced Usage
```bash
# Use specific camera
python main.py --camera 1

# Use custom model
python main.py --model path/to/model.h5

# Disable debug information
python main.py --no-debug
```

### Controls
- **Q/ESC**: Quit application
- **D**: Toggle debug information display
- **C**: Toggle confidence bars
- **E**: Toggle emotion dashboard
- **S**: Save screenshot
- **R**: Toggle video recording

## System Architecture

### Core Components

1. **Emotion Recognition Model** (`src/emotion_model.py`)
   - CNN-based architecture for 7-class emotion classification
   - Real-time inference optimization
   - Prediction smoothing and filtering

2. **Face Detection** (`src/face_detector.py`)
   - MediaPipe-based face detection
   - OpenCV fallback for robustness
   - Multi-face tracking support

3. **AR Overlay System** (`src/ar_overlay.py`)
   - Emoji rendering and positioning
   - Real-time overlay compositing
   - Confidence visualization

4. **Camera Processing** (`src/camera_processor.py`)
   - Optimized camera input handling
   - Frame rate optimization
   - Performance monitoring

5. **Utilities** (`utils/helpers.py`)
   - Analytics and reporting
   - Configuration management
   - System validation

### Data Flow
```
Camera Input → Face Detection → Face ROI Extraction → 
Emotion Prediction → Prediction Smoothing → AR Overlay → 
Display Output
```

## Performance Optimization

The system includes several optimization features:

- **Frame skipping**: Process every nth frame for better performance
- **Face quality assessment**: Skip low-quality detections
- **Prediction smoothing**: Reduce jitter through temporal filtering
- **Efficient AR rendering**: Optimized overlay compositing
- **Multi-threading support**: Parallel processing capabilities

## Emotion Recognition

### Supported Emotions
- 😠 Angry
- 🤢 Disgust  
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

### Model Architecture
The CNN model uses:
- Multiple convolutional layers with batch normalization
- MaxPooling for feature reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax activation for probability output

## Configuration

Key settings can be modified in `config.py`:

```python
# Model settings
MODEL_INPUT_SIZE = (48, 48)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# AR overlay settings
EMOJI_SIZE = 60
EMOJI_OFFSET_Y = -80
OVERLAY_ALPHA = 0.8

# Performance settings
SKIP_FRAMES = 2
MAX_FACES = 5
```

## Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure no other applications are using the camera

2. **Poor emotion recognition accuracy**
   - Ensure good lighting conditions
   - Position face clearly in camera view
   - Check if face is not too small or too far

3. **Low FPS performance**
   - Reduce camera resolution in config
   - Increase SKIP_FRAMES value
   - Disable debug overlays
   - Use GPU acceleration if available

## Data Source

The emotion recognition model can be trained using the FER2013 dataset:
- **Source**: https://www.kaggle.com/datasets/xavier00/fer2013-facial-expression-recognition-dataset
- **Format**: 48x48 grayscale images
- **Classes**: 7 emotion categories
- **Size**: ~35,000 images

## Privacy and Ethics

This system is designed with privacy considerations:
- No data is stored or transmitted externally
- All processing is performed locally
- Users have full control over when the system is active
- Optional screenshot saving requires explicit user action

## License

This project is licensed under the MIT License. See LICENSE file for details.
