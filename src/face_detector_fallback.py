"""
Face Detection and Tracking System (Fallback Version)
Uses OpenCV when MediaPipe is not available
"""

import cv2
import numpy as np
from config import (
    FACE_DETECTION_CONFIDENCE, 
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_FACES
)

class FaceDetector:
    def __init__(self):
        # Try to import MediaPipe, fallback to OpenCV only
        self.use_mediapipe = False
        self.mp_face_detection = None
        self.mp_drawing = None
        self.face_detection = None
        
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Face detection model
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range (2 meters), 1 for full-range (5 meters)
                min_detection_confidence=MIN_DETECTION_CONFIDENCE
            )
            self.use_mediapipe = True
            print("MediaPipe face detection initialized")
            
        except ImportError:
            print("MediaPipe not available, using OpenCV fallback")
        
        # OpenCV face detector (always available as backup)
        self.cv_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Face tracking variables
        self.face_trackers = []
        self.face_ids = []
        self.next_face_id = 0
        
    def detect_faces(self, frame):
        """
        Detect faces in frame using MediaPipe or OpenCV fallback
        Returns list of face bounding boxes and landmarks
        """
        faces = []
        
        if self.use_mediapipe and self.face_detection:
            faces = self._detect_faces_mediapipe(frame)
        
        # Use OpenCV as primary or fallback method
        if not faces:
            faces = self._detect_faces_opencv(frame)
            
        return faces
    
    def _detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe"""
        faces = []
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections[:MAX_FACES]:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Get confidence score
                confidence = detection.score[0] if detection.score else 0.5
                
                face_info = {
                    'bbox': (x, y, width, height),
                    'confidence': confidence,
                    'landmarks': None  # Could extract key points if needed
                }
                
                faces.append(face_info)
        
        return faces
    
    def _detect_faces_opencv(self, frame):
        """Detect faces using OpenCV Haar Cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_rects = self.cv_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in face_rects[:MAX_FACES]:
            face_info = {
                'bbox': (x, y, w, h),
                'confidence': 0.8,  # Default confidence for OpenCV
                'landmarks': None
            }
            faces.append(face_info)
            
        return faces
    
    def extract_face_roi(self, frame, bbox, padding=0.2):
        """
        Extract face region of interest with optional padding
        Returns cropped face image
        """
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate padded coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        # Extract face ROI
        face_roi = frame[y1:y2, x1:x2]
        
        return face_roi, (x1, y1, x2-x1, y2-y1)
    
    def get_face_center(self, bbox):
        """Get center point of face bounding box"""
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        return (center_x, center_y)
    
    def is_face_valid(self, bbox, min_size=(30, 30)):
        """Check if detected face meets minimum size requirements"""
        x, y, w, h = bbox
        return w >= min_size[0] and h >= min_size[1]
    
    def filter_overlapping_faces(self, faces, overlap_threshold=0.5):
        """Remove overlapping face detections using Non-Maximum Suppression"""
        if len(faces) <= 1:
            return faces
        
        # Convert to format needed for NMS
        boxes = []
        scores = []
        
        for face in faces:
            x, y, w, h = face['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(face['confidence'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            FACE_DETECTION_CONFIDENCE, 
            overlap_threshold
        )
        
        # Filter faces based on NMS results
        filtered_faces = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                filtered_faces.append(faces[i])
        
        return filtered_faces
    
    def cleanup(self):
        """Cleanup resources"""
        if self.use_mediapipe and hasattr(self, 'face_detection') and self.face_detection:
            self.face_detection.close()