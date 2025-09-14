import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

class FER2013DataLoader:
    """
    Data loader for FER2013 facial expression recognition dataset.
    
    The dataset contains grayscale 48x48 pixel face images with emotion labels:
    0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    """
    
    def __init__(self, data_path="data/fer2013.csv"):
        self.data_path = data_path
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.num_classes = len(self.emotions)
        
    def load_data(self):
        """Load and preprocess FER2013 dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Please download from Kaggle.")
            
        print("Loading FER2013 dataset...")
        df = pd.read_csv(self.data_path)
        
        # Extract pixel data and convert to numpy arrays
        pixels = df['pixels'].tolist()
        faces = []
        
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(48, 48)
            faces.append(face)
            
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)  # Add channel dimension
        
        # Normalize pixel values to [0, 1]
        faces = faces.astype('float32') / 255.0
        
        # Extract emotion labels
        emotions = pd.get_dummies(df['emotion']).values
        
        print(f"Dataset loaded: {faces.shape[0]} samples")
        print(f"Image shape: {faces.shape[1:]}")
        print(f"Number of classes: {emotions.shape[1]}")
        
        return faces, emotions
    
    def split_data(self, faces, emotions, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets."""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            faces, emotions, test_size=test_size, random_state=42, stratify=emotions
        )
        
        # Second split: separate train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples") 
        print(f"Test set: {X_test.shape[0]} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def preprocess_image(self, image):
        """Preprocess a single image for prediction."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image, (48, 48))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        
        return image
    
    def download_dataset(self):
        """Download FER2013 dataset from Kaggle using Kaggle API."""
        import kaggle
        
        try:
            print("Downloading FER2013 dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                'xavier00/fer2013-facial-expression-recognition-dataset',
                path='data/',
                unzip=True
            )
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please manually download from: https://www.kaggle.com/datasets/xavier00/fer2013-facial-expression-recognition-dataset")
            print("And place fer2013.csv in the data/ directory")

if __name__ == "__main__":
    # Example usage
    loader = FER2013DataLoader()
    
    # Download dataset if it doesn't exist
    if not os.path.exists(loader.data_path):
        loader.download_dataset()
    
    # Load and split data
    try:
        faces, emotions = loader.load_data()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.split_data(faces, emotions)
        
        print("\nDataset statistics:")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Emotion classes: {loader.emotions}")
        
    except FileNotFoundError as e:
        print(e)