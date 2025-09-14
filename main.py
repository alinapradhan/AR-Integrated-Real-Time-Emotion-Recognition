#!/usr/bin/env python3
"""
AR-Integrated Real-Time Emotion Recognition

This script provides a command-line interface for the emotion recognition system.
It supports training, real-time detection, and AR overlay modes.

Usage:
    python main.py train              # Train the emotion recognition model
    python main.py detect             # Run real-time emotion detection  
    python main.py ar                 # Run AR emotion overlay
    python main.py --help             # Show help message

Requirements:
    - FER2013 dataset (download from Kaggle)
    - Webcam for real-time detection
    - Python dependencies (see requirements.txt)
"""

import argparse
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def train_mode():
    """Train the emotion recognition model."""
    print("Starting model training...")
    from train_model import main as train_main
    train_main()

def detect_mode():
    """Run real-time emotion detection."""
    print("Starting real-time emotion detection...")
    from src.real_time_detector import main as detect_main
    detect_main()

def ar_mode():
    """Run AR emotion overlay."""
    print("Starting AR emotion overlay...")
    from src.ar_emotion_overlay import main as ar_main
    ar_main()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AR-Integrated Real-Time Emotion Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py train    # Train the model with FER2013 dataset
    python main.py detect   # Run real-time emotion detection
    python main.py ar       # Run AR emotion overlay
    
Before running, make sure to:
1. Install dependencies: pip install -r requirements.txt
2. Download FER2013 dataset and place in data/ directory
3. For webcam modes, ensure camera is connected and working
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'detect', 'ar'],
        help='Operation mode: train model, run detection, or AR overlay'
    )
    
    args = parser.parse_args()
    
    # Check if required directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        if args.mode == 'train':
            train_mode()
        elif args.mode == 'detect':
            detect_mode()
        elif args.mode == 'ar':
            ar_mode()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()