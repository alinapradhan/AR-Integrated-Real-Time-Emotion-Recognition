#!/usr/bin/env python3
"""
AR-Integrated Real-Time Emotion Recognition System
Main application entry point
"""

import cv2
import numpy as np
import sys
import os
import time
import argparse
from collections import deque

# Add src and utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from emotion_model import EmotionRecognitionModel
from face_detector import FaceDetector
from ar_overlay import AROverlay
from camera_processor import CameraProcessor, PerformanceMonitor
from helpers import (
    setup_logging, 
    EmotionAnalytics, 
    smooth_predictions,
    get_dominant_emotion,
    calculate_face_quality_score,
    validate_system_requirements
)
from config import *

class EmotionRecognitionApp:
    def __init__(self, camera_source=0, model_path=None):
        self.camera_source = camera_source
        self.model_path = model_path
        
        # Initialize components
        self.emotion_model = None
        self.face_detector = None
        self.ar_overlay = None
        self.camera_processor = None
        self.performance_monitor = PerformanceMonitor()
        self.analytics = EmotionAnalytics()
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        self.emotion_history = deque(maxlen=20)
        
        # Application state
        self.is_running = False
        self.show_debug_info = True
        self.show_confidence_bars = True
        self.show_emotion_dashboard = True
        self.record_video = False
        
        # Setup logging
        self.logger = setup_logging()
        
    def initialize_system(self):
        """Initialize all system components"""
        self.logger.info("Initializing AR Emotion Recognition System...")
        
        # Validate system requirements
        requirements = validate_system_requirements()
        missing_deps = [dep for dep, available in requirements.items() if not available]
        
        if missing_deps:
            self.logger.error(f"Missing dependencies: {missing_deps}")
            return False
        
        try:
            # Initialize emotion recognition model
            self.emotion_model = EmotionRecognitionModel(self.model_path)
            if not self.emotion_model.load_model():
                self.logger.error("Failed to load emotion recognition model")
                return False
            self.logger.info("Emotion model loaded successfully")
            
            # Initialize face detector
            self.face_detector = FaceDetector()
            self.logger.info("Face detector initialized")
            
            # Initialize AR overlay system
            self.ar_overlay = AROverlay()
            self.logger.info("AR overlay system initialized")
            
            # Initialize camera processor
            self.camera_processor = CameraProcessor(
                source=self.camera_source,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT
            )
            
            if not self.camera_processor.start_capture():
                self.logger.error("Failed to initialize camera")
                return False
            self.logger.info("Camera initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame for emotion recognition"""
        if frame is None:
            return None
        
        # Start timing
        processing_start = self.performance_monitor.start_frame_timing()
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        faces = self.face_detector.filter_overlapping_faces(faces)
        
        # Process each detected face
        processed_faces = []
        
        for face_info in faces:
            bbox = face_info['bbox']
            confidence = face_info['confidence']
            
            # Validate face
            if not self.face_detector.is_face_valid(bbox):
                continue
            
            # Extract face ROI
            face_roi, padded_bbox = self.face_detector.extract_face_roi(frame, bbox)
            
            # Calculate face quality
            quality_score = calculate_face_quality_score(face_roi)
            if quality_score < 0.3:  # Skip low quality faces
                continue
            
            try:
                # Predict emotion
                emotion, emotion_confidence, probabilities = self.emotion_model.predict_emotion(face_roi)
                
                # Smooth predictions
                self.prediction_history.append(probabilities)
                if len(self.prediction_history) > 1:
                    smoothed_probs = smooth_predictions(list(self.prediction_history))
                    emotion, emotion_confidence = get_dominant_emotion(
                        smoothed_probs, EMOTION_LABELS, min_confidence=0.3
                    )
                
                # Update emotion history
                self.emotion_history.append(emotion)
                
                # Update analytics
                self.analytics.update(emotion, emotion_confidence)
                
                # Get face center for AR overlay
                face_center = self.face_detector.get_face_center(bbox)
                
                processed_faces.append({
                    'bbox': bbox,
                    'emotion': emotion,
                    'confidence': emotion_confidence,
                    'probabilities': probabilities,
                    'center': face_center,
                    'quality': quality_score
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing face: {e}")
                continue
        
        # Update performance monitoring
        self.performance_monitor.end_frame_timing(processing_start)
        self.performance_monitor.increment_frame_count()
        self.analytics.increment_frame_count()
        
        return processed_faces
    
    def render_frame(self, frame, processed_faces):
        """Render frame with AR overlays and information"""
        if frame is None:
            return None
        
        rendered_frame = frame.copy()
        
        # Render each processed face
        for face_data in processed_faces:
            bbox = face_data['bbox']
            emotion = face_data['emotion']
            confidence = face_data['confidence']
            probabilities = face_data['probabilities']
            center = face_data['center']
            
            # Draw AR emoji overlay
            rendered_frame = self.ar_overlay.overlay_emoji(
                rendered_frame, emotion, center, confidence
            )
            
            # Draw debug information if enabled
            if self.show_debug_info:
                rendered_frame = self.ar_overlay.draw_face_info(
                    rendered_frame, bbox, emotion, confidence
                )
            
            # Draw confidence bar if enabled
            if self.show_confidence_bars:
                rendered_frame = self.ar_overlay.draw_confidence_bar(
                    rendered_frame, bbox, confidence
                )
            
            # Draw emotion dashboard if enabled
            if self.show_emotion_dashboard and len(processed_faces) == 1:
                rendered_frame = self.ar_overlay.create_emotion_dashboard(
                    rendered_frame, probabilities, EMOTION_LABELS
                )
        
        # Add FPS counter
        fps = self.camera_processor.get_fps()
        rendered_frame = self.ar_overlay.add_fps_counter(rendered_frame, fps)
        
        # Add face count
        cv2.putText(rendered_frame, f"Faces: {len(processed_faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return rendered_frame
    
    def handle_keyboard_input(self, key):
        """Handle keyboard input for controlling the application"""
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        elif key == ord('d'):  # Toggle debug info
            self.show_debug_info = not self.show_debug_info
            self.logger.info(f"Debug info: {'ON' if self.show_debug_info else 'OFF'}")
        elif key == ord('c'):  # Toggle confidence bars
            self.show_confidence_bars = not self.show_confidence_bars
            self.logger.info(f"Confidence bars: {'ON' if self.show_confidence_bars else 'OFF'}")
        elif key == ord('e'):  # Toggle emotion dashboard
            self.show_emotion_dashboard = not self.show_emotion_dashboard
            self.logger.info(f"Emotion dashboard: {'ON' if self.show_emotion_dashboard else 'OFF'}")
        elif key == ord('s'):  # Save screenshot
            # This would be implemented in the main loop
            pass
        elif key == ord('r'):  # Toggle recording
            self.record_video = not self.record_video
            self.logger.info(f"Recording: {'ON' if self.record_video else 'OFF'}")
        
        return True
    
    def run(self):
        """Main application loop"""
        if not self.initialize_system():
            self.logger.error("Failed to initialize system")
            return False
        
        self.is_running = True
        self.logger.info("Starting emotion recognition system...")
        
        # Display controls
        print("\n" + "="*50)
        print("AR EMOTION RECOGNITION SYSTEM - CONTROLS")
        print("="*50)
        print("Q/ESC : Quit application")
        print("D     : Toggle debug information")
        print("C     : Toggle confidence bars")
        print("E     : Toggle emotion dashboard")
        print("S     : Save screenshot")
        print("R     : Toggle video recording")
        print("="*50 + "\n")
        
        try:
            while self.is_running:
                # Read frame from camera
                frame, ret = self.camera_processor.read_frame()
                
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    break
                
                # Mirror the frame for better user experience
                frame = self.camera_processor.flip_frame(frame, horizontal=True)
                
                # Process frame for emotion recognition
                if self.camera_processor.should_process_frame():
                    processed_faces = self.process_frame(frame)
                else:
                    processed_faces = []  # Skip processing for performance
                
                # Render frame with AR overlays
                rendered_frame = self.render_frame(frame, processed_faces)
                
                # Display frame
                cv2.imshow('AR Emotion Recognition', rendered_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        # Stop camera
        if self.camera_processor:
            self.camera_processor.stop_capture()
        
        # Cleanup face detector
        if self.face_detector:
            self.face_detector.cleanup()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Print analytics summary
        summary = self.analytics.get_analytics_summary()
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Duration: {summary['session_duration']:.1f} seconds")
        print(f"Faces detected: {summary['total_faces_detected']}")
        print(f"Frames processed: {summary['total_frames_processed']}")
        print(f"Average confidence: {summary['average_confidence']:.2f}")
        print(f"Detection rate: {summary['detection_rate']:.2f}")
        print("Dominant emotions:")
        for emotion, count in summary['dominant_emotions']:
            print(f"  {emotion}: {count}")
        print("="*50)
        
        self.logger.info("Cleanup completed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AR-Integrated Real-Time Emotion Recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera source index')
    parser.add_argument('--model', type=str, help='Path to emotion recognition model')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug information')
    
    args = parser.parse_args()
    
    # Create and run application
    app = EmotionRecognitionApp(camera_source=args.camera, model_path=args.model)
    
    if args.no_debug:
        app.show_debug_info = False
    
    return app.run()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)