"""
Configuration file for AR Emotion Recognition System
Contains all constants and settings for the application
"""

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emoji mapping for each emotion
EMOTION_EMOJIS = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨', 
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲'
}

# Model configuration
MODEL_INPUT_SIZE = (48, 48)
MODEL_PATH = 'models/emotion_model.h5'
MODEL_WEIGHTS_PATH = 'models/emotion_weights.h5'

# Camera configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face detection configuration
FACE_DETECTION_CONFIDENCE = 0.7
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# AR overlay configuration
EMOJI_SIZE = 60
EMOJI_OFFSET_Y = -80  # Pixels above the face
EMOJI_OFFSET_X = 0    # Center aligned
OVERLAY_ALPHA = 0.8

# Performance optimization
SKIP_FRAMES = 2  # Process every nth frame
MAX_FACES = 5    # Maximum number of faces to track

# Colors (BGR format)
FACE_BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
CONFIDENCE_BAR_COLOR = (0, 255, 255)  # Yellow

# Font settings
FONT_SCALE = 0.7
FONT_THICKNESS = 2