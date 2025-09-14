import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from src.data_loader import FER2013DataLoader
import os

class RealTimeEmotionDetector:
    """
    Real-time emotion detection using webcam and trained CNN model.
    Integrates with MediaPipe for face detection and emotion recognition.
    """
    
    def __init__(self, model_path='models/emotion_model.h5'):
        self.model_path = model_path
        self.model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Emotion colors for visualization
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 0),    # Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 255),    # Yellow
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (255, 165, 0), # Orange
            'Neutral': (128, 128, 128) # Gray
        }
        
        self.data_loader = FER2013DataLoader()
        
    def load_model(self):
        """Load pre-trained emotion recognition model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train the model first.")
        
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
    
    def predict_emotion(self, face_roi):
        """Predict emotion for a detected face region."""
        if self.model is None:
            self.load_model()
        
        # Preprocess the face region
        processed_face = self.data_loader.preprocess_image(face_roi)
        
        # Make prediction
        prediction = self.model.predict(processed_face, verbose=0)
        emotion_idx = np.argmax(prediction)
        confidence = prediction[0][emotion_idx]
        emotion = self.emotion_labels[emotion_idx]
        
        return emotion, confidence
    
    def draw_emotion_overlay(self, image, emotion, confidence, bbox):
        """Draw emotion text and bounding box on the image."""
        x, y, w, h = bbox
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Prepare text
        text = f"{emotion}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Calculate text size for background
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(
            image, 
            (x, y - text_height - 10), 
            (x + text_width, y), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            image, 
            text, 
            (x, y - 5), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness
        )
        
        return image
    
    def process_frame(self, frame):
        """Process a single frame for emotion detection."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Ensure bounding box is within frame
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    # Predict emotion
                    emotion, confidence = self.predict_emotion(face_roi)
                    
                    # Draw overlay
                    frame = self.draw_emotion_overlay(frame, emotion, confidence, (x, y, w, h))
        
        return frame
    
    def run_real_time_detection(self):
        """Run real-time emotion detection using webcam."""
        print("Starting real-time emotion detection...")
        print("Press 'q' to quit")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame for emotion detection
                processed_frame = self.process_frame(frame)
                
                # Add instructions
                cv2.putText(
                    processed_frame, 
                    "Press 'q' to quit", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                
                # Display frame
                cv2.imshow('Real-Time Emotion Detection', processed_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Real-time detection stopped")

def main():
    """Main function to run real-time emotion detection."""
    detector = RealTimeEmotionDetector()
    
    try:
        detector.run_real_time_detection()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python train_model.py")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()