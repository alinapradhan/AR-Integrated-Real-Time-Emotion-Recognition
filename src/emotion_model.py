"""
Emotion Recognition Model
CNN-based model for detecting emotions from facial expressions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from config import MODEL_INPUT_SIZE, EMOTION_LABELS
import os

class EmotionRecognitionModel:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.input_size = MODEL_INPUT_SIZE
        self.emotion_labels = EMOTION_LABELS
        
    def build_model(self):
        """Build CNN model for emotion recognition"""
        model = keras.Sequential([
            # First Convolution Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.input_size, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolution Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolution Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.emotion_labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_model(self, model_path=None):
        """Load pre-trained model"""
        if model_path is None:
            model_path = self.model_path
            
        if model_path and os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                print(f"Model loaded from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            # Create and use a pre-trained model architecture
            print("No pre-trained model found. Creating new model...")
            self.model = self.build_model()
            self._load_pretrained_weights()
            return True
    
    def _load_pretrained_weights(self):
        """Load pre-trained weights or initialize with transfer learning approach"""
        # For demo purposes, we'll create a simple baseline model
        # In a real implementation, you would train this on FER2013 dataset
        print("Initializing model with baseline weights...")
        
        # Create some sample training data for demonstration
        # This would be replaced with actual FER2013 dataset training
        sample_data = np.random.random((100, *self.input_size, 1))
        sample_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, len(self.emotion_labels), 100), 
            len(self.emotion_labels)
        )
        
        # Quick training for demonstration
        self.model.fit(sample_data, sample_labels, epochs=1, verbose=0)
        print("Model initialized with baseline weights")
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model prediction"""
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        face_img = cv2.resize(face_img, self.input_size)
        
        # Normalize pixel values
        face_img = face_img.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess the face image
        processed_face = self.preprocess_face(face_img)
        
        # Get prediction
        prediction = self.model.predict(processed_face, verbose=0)
        
        # Get emotion label and confidence
        emotion_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][emotion_idx])
        emotion_label = self.emotion_labels[emotion_idx]
        
        return emotion_label, confidence, prediction[0]
    
    def save_model(self, path):
        """Save the trained model"""
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")