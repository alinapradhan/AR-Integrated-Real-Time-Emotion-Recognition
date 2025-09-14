import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from src.data_loader import FER2013DataLoader
import os

class AREmotionOverlay:
    """
    Augmented Reality emotion overlay system.
    Displays emotion-based emojis and graphics above detected faces.
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
        
        # Emoji representations for each emotion
        self.emotion_emojis = {
            'Angry': '😠',
            'Disgust': '🤢', 
            'Fear': '😨',
            'Happy': '😊',
            'Sad': '😢',
            'Surprise': '😲',
            'Neutral': '😐'
        }
        
        # Colors for AR overlays
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
    
    def create_emotion_particle_effect(self, emotion, frame_shape):
        """Create particle effect based on emotion."""
        particles = []
        
        if emotion == 'Happy':
            # Generate sparkles/stars for happiness
            for _ in range(10):
                particles.append({
                    'type': 'star',
                    'color': (0, 255, 255),
                    'size': np.random.randint(3, 8)
                })
        elif emotion == 'Angry':
            # Generate fire-like particles for anger
            for _ in range(8):
                particles.append({
                    'type': 'fire',
                    'color': (0, 0, 255),
                    'size': np.random.randint(4, 10)
                })
        elif emotion == 'Sad':
            # Generate tear drops for sadness
            for _ in range(5):
                particles.append({
                    'type': 'tear',
                    'color': (255, 200, 200),
                    'size': np.random.randint(2, 6)
                })
        
        return particles
    
    def draw_ar_overlay(self, image, emotion, confidence, bbox):
        """Draw AR overlay with emoji and effects above detected face."""
        x, y, w, h = bbox
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Calculate center point above the face
        center_x = x + w // 2
        overlay_y = max(y - 80, 50)  # Position above face
        
        # Draw emotion indicator circle
        circle_radius = 40
        cv2.circle(image, (center_x, overlay_y), circle_radius, color, 3)
        cv2.circle(image, (center_x, overlay_y), circle_radius - 5, (255, 255, 255), -1)
        
        # Add emoji text (simplified as text since OpenCV doesn't support emojis directly)
        emoji_text = self.emotion_emojis.get(emotion, '?')
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw large emotion symbol
        cv2.putText(
            image,
            emotion[:3].upper(),  # First 3 letters of emotion
            (center_x - 15, overlay_y + 5),
            font,
            0.8,
            color,
            2
        )
        
        # Draw confidence bar
        bar_width = 80
        bar_height = 8
        bar_x = center_x - bar_width // 2
        bar_y = overlay_y + 60
        
        # Background bar
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (100, 100, 100),
            -1
        )
        
        # Confidence bar
        confidence_width = int(bar_width * confidence)
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + confidence_width, bar_y + bar_height),
            color,
            -1
        )
        
        # Confidence text
        cv2.putText(
            image,
            f"{confidence:.0%}",
            (bar_x, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )
        
        # Draw emotion name
        cv2.putText(
            image,
            emotion,
            (center_x - len(emotion) * 8, overlay_y - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Add AR particles/effects
        particles = self.create_emotion_particle_effect(emotion, image.shape)
        for particle in particles:
            # Random position around the face
            px = center_x + np.random.randint(-60, 60)
            py = overlay_y + np.random.randint(-40, 40)
            
            if particle['type'] == 'star':
                # Draw simple star shape
                cv2.circle(image, (px, py), particle['size'], particle['color'], -1)
            elif particle['type'] == 'fire':
                # Draw flame-like shape
                cv2.circle(image, (px, py), particle['size'], particle['color'], -1)
                cv2.circle(image, (px, py-3), particle['size']//2, (0, 100, 255), -1)
            elif particle['type'] == 'tear':
                # Draw tear drop
                cv2.ellipse(image, (px, py), (particle['size'], particle['size']*2), 0, 0, 360, particle['color'], -1)
        
        # Draw face bounding box with pulsing effect
        pulse_thickness = int(3 + 2 * np.sin(cv2.getTickCount() * 0.01))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, pulse_thickness)
        
        return image
    
    def process_frame(self, frame):
        """Process a single frame for AR emotion overlay."""
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
                    
                    # Draw AR overlay
                    frame = self.draw_ar_overlay(frame, emotion, confidence, (x, y, w, h))
        
        return frame
    
    def run_ar_emotion_overlay(self):
        """Run AR emotion overlay using webcam."""
        print("Starting AR Emotion Overlay...")
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
                
                # Process frame for AR emotion overlay
                processed_frame = self.process_frame(frame)
                
                # Add title and instructions
                cv2.putText(
                    processed_frame,
                    "AR Emotion Recognition",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                
                cv2.putText(
                    processed_frame,
                    "Press 'q' to quit",
                    (10, processed_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # Display frame
                cv2.imshow('AR Emotion Overlay', processed_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("AR emotion overlay stopped")

def main():
    """Main function to run AR emotion overlay."""
    ar_overlay = AREmotionOverlay()
    
    try:
        ar_overlay.run_ar_emotion_overlay()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python train_model.py")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()