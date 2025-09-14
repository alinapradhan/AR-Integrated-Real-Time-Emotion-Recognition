import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import FER2013DataLoader
from src.emotion_model import EmotionCNN
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    """Plot training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

def main():
    """Main training script for emotion recognition model."""
    print("=" * 50)
    print("FER2013 Emotion Recognition Model Training")
    print("=" * 50)
    
    # Initialize data loader
    data_loader = FER2013DataLoader()
    
    # Check if dataset exists, if not provide instructions
    if not os.path.exists(data_loader.data_path):
        print("Dataset not found!")
        print("Please follow these steps:")
        print("1. Create a Kaggle account at https://www.kaggle.com/")
        print("2. Go to Account settings and create an API token")
        print("3. Place kaggle.json in ~/.kaggle/ directory")
        print("4. Download the dataset: https://www.kaggle.com/datasets/xavier00/fer2013-facial-expression-recognition-dataset")
        print("5. Place fer2013.csv in the data/ directory")
        
        # Try to download automatically
        try:
            data_loader.download_dataset()
        except Exception as e:
            print(f"Automatic download failed: {e}")
            print("Please download manually and place fer2013.csv in data/ directory")
            return
    
    try:
        # Load and prepare data
        print("\nLoading FER2013 dataset...")
        faces, emotions = data_loader.load_data()
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(faces, emotions)
        
        # Initialize and build model
        print("\nBuilding CNN model...")
        cnn = EmotionCNN()
        cnn.build_model()
        cnn.compile_model(learning_rate=0.001)
        
        print("\nModel Architecture:")
        cnn.summary()
        
        # Train model
        print("\nStarting training...")
        history = cnn.train(
            X_train, y_train, 
            X_val, y_val,
            epochs=50,
            batch_size=32
        )
        
        # Evaluate model
        print("\nEvaluating model on test set...")
        test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/emotion_model.h5'
        cnn.save_model(model_path)
        
        # Plot training history
        plot_training_history(history)
        
        # Test model with a few samples
        print("\nTesting model with sample predictions...")
        sample_indices = np.random.choice(len(X_test), 5, replace=False)
        
        for idx in sample_indices:
            sample_image = np.expand_dims(X_test[idx], axis=0)
            emotion, confidence = cnn.predict_emotion(sample_image, data_loader.emotions)
            actual_emotion = data_loader.emotions[np.argmax(y_test[idx])]
            
            print(f"Predicted: {emotion} ({confidence:.2f}) | Actual: {actual_emotion}")
        
        print(f"\nTraining completed successfully!")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        print(f"Model saved to: {model_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the FER2013 dataset is available in the data/ directory")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()