#!/usr/bin/env python3
"""
Demo script for AR Emotion Recognition System
This demo shows the system working with simulated data when TensorFlow is not available
"""

import cv2
import numpy as np
import sys
import os
import time
import random
from collections import deque

# Add src and utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from face_detector import FaceDetector
from ar_overlay import AROverlay
from camera_processor import CameraProcessor, PerformanceMonitor
from helpers import EmotionAnalytics
from config import *

class MockEmotionModel:
    """Mock emotion model for demo when TensorFlow is not available"""
    
    def __init__(self, model_path=None):
        self.emotion_labels = EMOTION_LABELS
        self.last_predictions = {}  # Store per face for consistency
        
    def load_model(self):
        print("Mock emotion model loaded (TensorFlow not available)")
        return True
    
    def predict_emotion(self, face_img):
        """Mock prediction - returns random emotion with some temporal consistency"""
        # Create a simple hash of the face region for consistency
        face_hash = hash(face_img.tobytes() if face_img is not None else b'')
        
        # Use some consistency in predictions
        if face_hash in self.last_predictions:
            # 70% chance to keep the same emotion for stability
            if random.random() < 0.7:
                emotion, confidence, probabilities = self.last_predictions[face_hash]
                # Slightly vary the confidence
                confidence = max(0.5, min(1.0, confidence + random.uniform(-0.1, 0.1)))
                return emotion, confidence, probabilities
        
        # Generate new prediction
        emotion_idx = random.randint(0, len(self.emotion_labels) - 1)
        emotion = self.emotion_labels[emotion_idx]
        confidence = random.uniform(0.6, 0.95)
        
        # Create probability distribution
        probabilities = np.random.random(len(self.emotion_labels))
        probabilities[emotion_idx] = confidence
        probabilities = probabilities / probabilities.sum() * confidence
        probabilities[emotion_idx] = confidence
        
        # Store for consistency
        self.last_predictions[face_hash] = (emotion, confidence, probabilities)
        
        return emotion, confidence, probabilities

def demo_with_static_image():
    """Demo using a static test image"""
    print("Running AR Emotion Recognition Demo with Static Image")
    print("="*60)
    
    # Create a test image with simulated face
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image.fill(50)  # Dark background
    
    # Draw a simple face-like rectangle
    cv2.rectangle(test_image, (200, 150), (440, 350), (100, 100, 100), -1)  # Face area
    cv2.circle(test_image, (280, 220), 15, (255, 255, 255), -1)  # Left eye
    cv2.circle(test_image, (360, 220), 15, (255, 255, 255), -1)  # Right eye
    cv2.ellipse(test_image, (320, 280), (40, 20), 0, 0, 180, (255, 255, 255), 2)  # Mouth
    
    # Add text
    cv2.putText(test_image, "AR Emotion Recognition Demo", (150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(test_image, "Press 'q' to quit, 'space' for new emotion", (130, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Initialize components
    face_detector = FaceDetector()
    ar_overlay = AROverlay()
    mock_model = MockEmotionModel()
    mock_model.load_model()
    analytics = EmotionAnalytics()
    
    print("Demo initialized successfully!")
    print("\nControls:")
    print("- Press SPACE to generate new emotion")
    print("- Press 'q' or ESC to quit")
    print("- Press 'd' to toggle debug info")
    print("- Press 'c' to toggle confidence bars")
    
    # Demo settings
    show_debug = True
    show_confidence = True
    current_emotion = "Happy"
    current_confidence = 0.85
    probabilities = np.array([0.1, 0.05, 0.08, 0.85, 0.1, 0.07, 0.06])  # Happy dominant
    
    # Simulated face bounding box
    face_bbox = (200, 150, 240, 200)  # x, y, w, h
    face_center = (320, 250)  # Center of the face
    
    while True:
        # Create display frame
        display_frame = test_image.copy()
        
        # Update analytics
        analytics.update(current_emotion, current_confidence)
        analytics.increment_frame_count()
        
        # Render AR overlay
        display_frame = ar_overlay.overlay_emoji(display_frame, current_emotion, face_center, current_confidence)
        
        # Render debug information
        if show_debug:
            display_frame = ar_overlay.draw_face_info(display_frame, face_bbox, current_emotion, current_confidence)
        
        # Render confidence bar
        if show_confidence:
            display_frame = ar_overlay.draw_confidence_bar(display_frame, face_bbox, current_confidence)
        
        # Render emotion dashboard
        display_frame = ar_overlay.create_emotion_dashboard(display_frame, probabilities, EMOTION_LABELS)
        
        # Add FPS (simulated)
        display_frame = ar_overlay.add_fps_counter(display_frame, 30.0)
        
        # Add face count
        cv2.putText(display_frame, "Faces: 1", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('AR Emotion Recognition Demo', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:  # Quit
            break
        elif key == ord(' '):  # Space - generate new emotion
            emotion_idx = random.randint(0, len(EMOTION_LABELS) - 1)
            current_emotion = EMOTION_LABELS[emotion_idx]
            current_confidence = random.uniform(0.6, 0.95)
            
            # Generate new probabilities
            probabilities = np.random.random(len(EMOTION_LABELS))
            probabilities = probabilities / probabilities.sum() * 0.8
            probabilities[emotion_idx] = current_confidence
            
            print(f"New emotion: {current_emotion} ({current_confidence:.2f})")
            
        elif key == ord('d'):  # Toggle debug
            show_debug = not show_debug
            print(f"Debug info: {'ON' if show_debug else 'OFF'}")
            
        elif key == ord('c'):  # Toggle confidence
            show_confidence = not show_confidence
            print(f"Confidence bars: {'ON' if show_confidence else 'OFF'}")
    
    # Cleanup
    cv2.destroyAllWindows()
    
    # Show analytics summary
    summary = analytics.get_analytics_summary()
    print("\n" + "="*50)
    print("DEMO SESSION SUMMARY")
    print("="*50)
    print(f"Duration: {summary['session_duration']:.1f} seconds")
    print(f"Total emotions shown: {summary['total_faces_detected']}")
    print(f"Frames processed: {summary['total_frames_processed']}")
    print("Emotion distribution:")
    for emotion, count in summary['emotion_distribution'].items():
        print(f"  {emotion}: {count}")
    print("="*50)

def test_face_detection():
    """Test face detection with webcam if available"""
    print("\nTesting Face Detection...")
    
    try:
        # Try to initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not available, skipping face detection test")
            return
        
        face_detector = FaceDetector()
        ar_overlay = AROverlay()
        
        print("Camera initialized. Testing face detection for 10 seconds...")
        print("Look at the camera to test face detection!")
        
        start_time = time.time()
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = face_detector.detect_faces(frame)
            faces = face_detector.filter_overlapping_faces(faces)
            
            # Draw face detection results
            for face_info in faces:
                bbox = face_info['bbox']
                confidence = face_info['confidence']
                
                # Draw bounding box
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw confidence
                cv2.putText(frame, f"Face: {confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add a demo emoji
                center = face_detector.get_face_center(bbox)
                frame = ar_overlay.overlay_emoji(frame, "Happy", center, confidence)
            
            # Add info text
            cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Testing face detection...", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Face Detection Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Face detection test completed!")
        
    except Exception as e:
        print(f"Face detection test failed: {e}")

def main():
    """Main demo function"""
    print("AR-Integrated Real-Time Emotion Recognition System")
    print("="*60)
    print("DEMO MODE - TensorFlow not available")
    print("This demo shows the system functionality with simulated data")
    print("="*60)
    
    # Check system components
    print("\nChecking system components...")
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not available")
        return
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy not available")
        return
    
    try:
        import config
        print("✓ Configuration loaded")
    except ImportError:
        print("✗ Configuration not available")
        return
    
    print("✓ All basic components available")
    
    # Run static image demo
    print("\n1. Running static image demo...")
    demo_with_static_image()
    
    # Test face detection if camera is available
    print("\n2. Testing face detection...")
    test_face_detection()
    
    print("\nDemo completed successfully!")
    print("For full functionality, install TensorFlow and run: python main.py")

if __name__ == "__main__":
    main()