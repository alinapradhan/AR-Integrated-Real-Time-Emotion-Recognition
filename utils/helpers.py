"""
Utility functions for the emotion recognition system
"""

import os
import numpy as np
import cv2
import json
from datetime import datetime
import logging

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('emotion_recognition.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directory_exists(directory_path):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return True
    return False

def load_config(config_path='config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default settings.")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in {config_path}. Using default settings.")
        return {}

def save_config(config, config_path='config.json'):
    """Save configuration to JSON file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def normalize_face_size(face_img, target_size=(48, 48)):
    """Normalize face image to consistent size"""
    if face_img is None:
        return None
    
    # Resize to target size
    normalized = cv2.resize(face_img, target_size)
    
    # Convert to grayscale if needed
    if len(normalized.shape) == 3:
        normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    
    return normalized

def calculate_face_quality_score(face_img):
    """Calculate quality score for face image based on clarity and size"""
    if face_img is None:
        return 0.0
    
    gray = face_img
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance (measure of blur)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize to 0-1 range (higher is better)
    blur_score = min(1.0, laplacian_var / 100.0)
    
    # Calculate size score
    h, w = gray.shape
    size_score = min(1.0, (h * w) / (100 * 100))
    
    # Calculate brightness score
    mean_brightness = np.mean(gray)
    brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
    
    # Combined quality score
    quality_score = (blur_score * 0.5 + size_score * 0.3 + brightness_score * 0.2)
    
    return quality_score

def smooth_predictions(predictions_history, window_size=5, weight_current=0.6):
    """Smooth emotion predictions over time to reduce jitter"""
    if len(predictions_history) < 2:
        return predictions_history[-1] if predictions_history else None
    
    # Take last window_size predictions
    recent_predictions = predictions_history[-window_size:]
    
    if len(recent_predictions) == 1:
        return recent_predictions[0]
    
    # Calculate weighted average
    current_pred = recent_predictions[-1]
    prev_preds = recent_predictions[:-1]
    
    # Weight current prediction more heavily
    smoothed = current_pred * weight_current
    
    # Add weighted average of previous predictions
    if prev_preds:
        prev_avg = np.mean(prev_preds, axis=0)
        smoothed += prev_avg * (1 - weight_current)
    
    return smoothed

def detect_emotion_change(current_emotion, previous_emotion, confidence_threshold=0.7):
    """Detect significant emotion changes"""
    if current_emotion != previous_emotion:
        return True
    return False

def get_dominant_emotion(emotion_probabilities, emotion_labels, min_confidence=0.3):
    """Get dominant emotion with confidence threshold"""
    max_idx = np.argmax(emotion_probabilities)
    max_confidence = emotion_probabilities[max_idx]
    
    if max_confidence >= min_confidence:
        return emotion_labels[max_idx], max_confidence
    else:
        return 'Neutral', max_confidence

def calculate_emotion_stability(emotion_history, window_size=10):
    """Calculate stability of emotion detection over time"""
    if len(emotion_history) < window_size:
        return 0.0
    
    recent_emotions = emotion_history[-window_size:]
    unique_emotions = set(recent_emotions)
    
    # Stability is inversely related to number of unique emotions
    stability = 1.0 - (len(unique_emotions) - 1) / (window_size - 1)
    
    return max(0.0, stability)

def save_screenshot(frame, emotion, confidence, save_dir='screenshots'):
    """Save screenshot of detected emotion"""
    ensure_directory_exists(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{emotion}_{confidence:.2f}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    try:
        cv2.imwrite(filepath, frame)
        return filepath
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return None

def create_emotion_report(emotion_data, output_path='emotion_report.json'):
    """Create detailed emotion recognition report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_faces_detected': emotion_data.get('total_faces', 0),
        'emotion_distribution': emotion_data.get('emotion_counts', {}),
        'average_confidence': emotion_data.get('avg_confidence', 0.0),
        'session_duration': emotion_data.get('duration', 0.0),
        'performance_metrics': emotion_data.get('performance', {})
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        return output_path
    except Exception as e:
        print(f"Error creating report: {e}")
        return None

def validate_system_requirements():
    """Validate system requirements and dependencies"""
    requirements = {
        'opencv': False,
        'tensorflow': False,
        'mediapipe': False,
        'numpy': False,
        'pillow': False
    }
    
    try:
        import cv2
        requirements['opencv'] = True
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        requirements['tensorflow'] = True
    except ImportError:
        pass
    
    try:
        import mediapipe
        requirements['mediapipe'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        requirements['numpy'] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        requirements['pillow'] = True
    except ImportError:
        pass
    
    return requirements

def benchmark_system_performance():
    """Benchmark system performance for optimization"""
    import time
    
    # Test basic operations
    results = {}
    
    # Test image processing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    for _ in range(100):
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
    results['image_processing'] = time.time() - start_time
    
    # Test array operations
    start_time = time.time()
    for _ in range(1000):
        arr = np.random.random((48, 48, 1))
        normalized = arr / 255.0
        expanded = np.expand_dims(normalized, axis=0)
    results['array_operations'] = time.time() - start_time
    
    return results

class EmotionAnalytics:
    def __init__(self):
        self.session_start_time = datetime.now()
        self.emotion_counts = {}
        self.confidence_scores = []
        self.face_count = 0
        self.total_frames = 0
        
    def update(self, emotion, confidence):
        """Update analytics with new emotion detection"""
        # Update emotion counts
        if emotion in self.emotion_counts:
            self.emotion_counts[emotion] += 1
        else:
            self.emotion_counts[emotion] = 1
        
        # Update confidence scores
        self.confidence_scores.append(confidence)
        self.face_count += 1
        
    def increment_frame_count(self):
        """Increment total frame counter"""
        self.total_frames += 1
        
    def get_session_duration(self):
        """Get current session duration in seconds"""
        return (datetime.now() - self.session_start_time).total_seconds()
    
    def get_dominant_emotions(self, top_n=3):
        """Get top N dominant emotions"""
        if not self.emotion_counts:
            return []
        
        sorted_emotions = sorted(self.emotion_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:top_n]
    
    def get_average_confidence(self):
        """Get average confidence score"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    def get_analytics_summary(self):
        """Get complete analytics summary"""
        return {
            'session_duration': self.get_session_duration(),
            'total_faces_detected': self.face_count,
            'total_frames_processed': self.total_frames,
            'emotion_distribution': self.emotion_counts,
            'average_confidence': self.get_average_confidence(),
            'dominant_emotions': self.get_dominant_emotions(),
            'detection_rate': self.face_count / max(1, self.total_frames)
        }