#!/usr/bin/env python3
"""
Example usage script for AR-Integrated Real-Time Emotion Recognition.
This script demonstrates how to use the system with the FER2013 dataset.
"""

import os
import sys
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def example_dataset_loading():
    """Example of loading and using FER2013 dataset."""
    print("📊 Example: Loading FER2013 Dataset")
    print("-" * 40)
    
    from data_loader import FER2013DataLoader
    
    # Initialize data loader
    loader = FER2013DataLoader()
    
    print(f"Dataset path: {loader.data_path}")
    print(f"Emotion classes: {loader.emotions}")
    print(f"Number of classes: {loader.num_classes}")
    
    # Check if dataset exists
    if not os.path.exists(loader.data_path):
        print("⚠️  Dataset not found. To download:")
        print("1. Set up Kaggle API credentials")
        print("2. Run: python setup.py")
        print("3. Or download manually from Kaggle")
        return False
    
    try:
        # Load dataset
        print("\n📥 Loading dataset...")
        faces, emotions = loader.load_data()
        
        print(f"✅ Loaded {faces.shape[0]} face images")
        print(f"✅ Image shape: {faces.shape[1:]}")
        print(f"✅ Emotion labels shape: {emotions.shape}")
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.split_data(faces, emotions)
        
        print(f"\n📊 Data splits:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples") 
        print(f"   Test: {X_test.shape[0]} samples")
        
        # Show emotion distribution
        print(f"\n😊 Emotion distribution in training set:")
        for i, emotion in enumerate(loader.emotions):
            count = np.sum(np.argmax(y_train, axis=1) == i)
            percentage = (count / len(y_train)) * 100
            print(f"   {emotion}: {count} samples ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def example_model_training():
    """Example of training emotion recognition model."""
    print("\n🧠 Example: Model Training")
    print("-" * 40)
    
    from emotion_model import EmotionCNN
    
    # Initialize model
    cnn = EmotionCNN(input_shape=(48, 48, 1), num_classes=7)
    
    # Build and compile model
    cnn.build_model()
    cnn.compile_model(learning_rate=0.001)
    
    print("✅ Model architecture:")
    cnn.summary()
    
    print(f"\n🔧 Training configuration:")
    print(f"   - Input shape: {cnn.input_shape}")
    print(f"   - Number of classes: {cnn.num_classes}")
    print(f"   - Optimizer: Adam")
    print(f"   - Loss: Categorical crossentropy")
    
    # Note: Actual training would require dataset
    print(f"\n📝 To train the model:")
    print(f"   python main.py train")
    
    return True

def example_real_time_detection():
    """Example of real-time emotion detection setup."""
    print("\n📹 Example: Real-time Detection")
    print("-" * 40)
    
    from real_time_detector import RealTimeEmotionDetector
    
    # Initialize detector
    detector = RealTimeEmotionDetector()
    
    print(f"Model path: {detector.model_path}")
    print(f"Emotion labels: {detector.emotion_labels}")
    print(f"Emotion colors: {list(detector.emotion_colors.keys())}")
    
    print(f"\n🎯 Features:")
    print(f"   - MediaPipe face detection")
    print(f"   - Real-time emotion classification")
    print(f"   - Bounding box visualization")
    print(f"   - Confidence display")
    
    print(f"\n📝 To run real-time detection:")
    print(f"   python main.py detect")
    
    return True

def example_ar_overlay():
    """Example of AR emotion overlay features."""
    print("\n✨ Example: AR Emotion Overlay")
    print("-" * 40)
    
    from ar_emotion_overlay import AREmotionOverlay
    
    # Initialize AR overlay
    ar_overlay = AREmotionOverlay()
    
    print(f"Emotion emojis: {ar_overlay.emotion_emojis}")
    print(f"AR colors: {list(ar_overlay.emotion_colors.keys())}")
    
    print(f"\n🎨 AR Features:")
    print(f"   - Emotion-based color coding")
    print(f"   - Dynamic particle effects")
    print(f"   - Confidence visualization")
    print(f"   - Interactive overlays")
    
    print(f"\n🎆 Particle effects:")
    print(f"   - Happy: Sparkles and stars")
    print(f"   - Angry: Fire-like particles")
    print(f"   - Sad: Tear drops")
    
    print(f"\n📝 To run AR overlay:")
    print(f"   python main.py ar")
    
    return True

def show_system_overview():
    """Show complete system overview."""
    print("\n🎭 System Overview")
    print("=" * 50)
    
    print("\n📋 Core Components:")
    print("   1. 📊 FER2013 Data Loader - Dataset handling and preprocessing")
    print("   2. 🧠 CNN Model - Deep learning architecture for emotion recognition")  
    print("   3. 📹 Real-time Detector - Live webcam emotion detection")
    print("   4. ✨ AR Overlay - Augmented reality visualization")
    
    print("\n🎯 Supported Emotions:")
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    for i, emotion in enumerate(emotions):
        print(f"   {i}: {emotion}")
    
    print("\n⚙️ Technical Specifications:")
    print("   - Input: 48x48 grayscale images")
    print("   - Model: Convolutional Neural Network")
    print("   - Face Detection: MediaPipe")
    print("   - Real-time Performance: Optimized for webcam")
    
    print("\n🚀 Usage Modes:")
    print("   - train: Train model on FER2013 dataset")
    print("   - detect: Real-time emotion detection")  
    print("   - ar: AR-enhanced emotion visualization")

def main():
    """Run all examples."""
    print("🎭 AR-Integrated Real-Time Emotion Recognition Examples")
    print("=" * 60)
    
    try:
        # Show system overview
        show_system_overview()
        
        # Run examples
        example_dataset_loading()
        example_model_training()
        example_real_time_detection()
        example_ar_overlay()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n📖 For complete setup instructions, run: python setup.py")
        print("📖 For usage help, run: python main.py --help")
        
    except ImportError as e:
        print(f"⚠️  Some modules not available: {e}")
        print("💡 Install dependencies with: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error running examples: {e}")

if __name__ == "__main__":
    main()