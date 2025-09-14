#!/usr/bin/env python3
"""
Setup script for AR-Integrated Real-Time Emotion Recognition system.
This script helps users set up the environment and download the FER2013 dataset.
"""

import os
import subprocess
import sys
import urllib.request
import json

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required. Current version:", sys.version)
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
        return True

def install_dependencies():
    """Install required Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install dependencies from requirements.txt
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install dependencies:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = ['data', 'models', 'config']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory}/ directory ready")
        except Exception as e:
            print(f"❌ Failed to create {directory}/ directory: {e}")
            return False
    
    return True

def setup_kaggle_api():
    """Help user set up Kaggle API for dataset download."""
    print("\n🔑 Setting up Kaggle API...")
    
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_json_path):
        print("✅ Kaggle API token found")
        return True
    else:
        print("⚠️  Kaggle API token not found")
        print("\nTo download the FER2013 dataset automatically, please:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to API section and click 'Create New API Token'")
        print("3. Download the kaggle.json file")
        print(f"4. Place it at: {kaggle_json_path}")
        print("5. Run this setup script again")
        
        # Create .kaggle directory
        try:
            os.makedirs(kaggle_dir, exist_ok=True)
            print(f"✅ Created directory: {kaggle_dir}")
        except Exception as e:
            print(f"❌ Failed to create Kaggle directory: {e}")
        
        return False

def download_fer2013_dataset():
    """Download FER2013 dataset using Kaggle API."""
    print("\n💾 Attempting to download FER2013 dataset...")
    
    try:
        # Try to import kaggle
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        dataset_name = 'xavier00/fer2013-facial-expression-recognition-dataset'
        print(f"📥 Downloading {dataset_name}...")
        
        api.dataset_download_files(dataset_name, path='data/', unzip=True)
        
        if os.path.exists('data/fer2013.csv'):
            print("✅ FER2013 dataset downloaded successfully!")
            
            # Show dataset info
            import pandas as pd
            df = pd.read_csv('data/fer2013.csv')
            print(f"📊 Dataset contains {len(df)} samples")
            print(f"📊 Emotion distribution:")
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion_counts = df['emotion'].value_counts().sort_index()
            for i, (emotion, count) in enumerate(zip(emotion_labels, emotion_counts)):
                print(f"   {emotion}: {count} samples")
            
            return True
        else:
            print("❌ Dataset download completed but fer2013.csv not found")
            return False
            
    except ImportError:
        print("❌ Kaggle module not available. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("\n📝 Manual download instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/xavier00/fer2013-facial-expression-recognition-dataset")
        print("2. Click 'Download' button")
        print("3. Extract fer2013.csv to the data/ directory")
        return False

def test_webcam():
    """Test if webcam is available for real-time detection."""
    print("\n📷 Testing webcam availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Webcam is working and ready for real-time detection")
                cap.release()
                return True
            else:
                print("⚠️  Webcam detected but cannot capture frames")
                cap.release()
                return False
        else:
            print("⚠️  No webcam detected")
            return False
    except ImportError:
        print("⚠️  OpenCV not available for webcam testing")
        return False
    except Exception as e:
        print(f"⚠️  Error testing webcam: {e}")
        return False

def show_next_steps():
    """Show next steps after setup."""
    print("\n🚀 Next Steps:")
    print("="*50)
    
    if os.path.exists('data/fer2013.csv'):
        print("1. 🎯 Train the emotion recognition model:")
        print("   python main.py train")
        print("\n2. 🎭 Run real-time emotion detection:")
        print("   python main.py detect")
        print("\n3. ✨ Run AR emotion overlay:")
        print("   python main.py ar")
    else:
        print("1. 📥 Download FER2013 dataset:")
        print("   - Set up Kaggle API (see instructions above)")
        print("   - Run this setup script again, or")
        print("   - Download manually from Kaggle")
        print("\n2. 🎯 Then train the model:")
        print("   python main.py train")
    
    print("\n📖 For more information, see README.md")

def main():
    """Main setup function."""
    print("🎭 AR-Integrated Real-Time Emotion Recognition Setup")
    print("="*60)
    
    success = True
    
    # Check Python version
    success &= check_python_version()
    
    # Set up directories
    success &= setup_directories()
    
    if success:
        print("\n✅ Basic setup completed!")
        
        # Try to install dependencies
        print("\nOptional steps (require internet connection):")
        install_success = install_dependencies()
        
        if install_success:
            # Set up Kaggle API
            kaggle_ready = setup_kaggle_api()
            
            if kaggle_ready:
                # Try to download dataset
                download_fer2013_dataset()
            
            # Test webcam
            test_webcam()
    
    # Show next steps
    show_next_steps()
    
    print("\n" + "="*60)
    if success:
        print("✅ Setup completed! The system is ready to use.")
    else:
        print("⚠️  Setup completed with some warnings. See above for details.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())