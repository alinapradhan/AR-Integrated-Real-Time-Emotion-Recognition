# AR-Integrated-Real-Time-Emotion-Recognition

This project integrates a CNN-based facial emotion recognition model with Augmented Reality to overlay live emotion feedback, such as emojis, above detected faces in real time. The system uses the FER2013 dataset for training and provides real-time emotion detection with AR visualizations.

## Features

- **Deep Learning Model**: CNN architecture trained on FER2013 dataset
- **Real-time Detection**: Live emotion recognition using webcam
- **AR Integration**: Augmented reality overlays with emotion-based visual effects
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Interactive Interface**: Command-line interface for different modes

## Dataset

**Data Source**: [FER2013 Facial Expression Recognition Dataset](https://www.kaggle.com/datasets/xavier00/fer2013-facial-expression-recognition-dataset)

The FER2013 dataset contains:
- 35,887 grayscale 48x48 pixel face images
- 7 emotion categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
- Pre-divided into training, validation, and test sets

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alinapradhan/AR-Integrated-Real-Time-Emotion-Recognition.git
   cd AR-Integrated-Real-Time-Emotion-Recognition
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download FER2013 Dataset**:
   - Create a Kaggle account at [kaggle.com](https://www.kaggle.com/)
   - Go to Account settings and create an API token
   - Place `kaggle.json` in `~/.kaggle/` directory
   - Download dataset: [FER2013 Dataset](https://www.kaggle.com/datasets/xavier00/fer2013-facial-expression-recognition-dataset)
   - Extract and place `fer2013.csv` in the `data/` directory

## Usage

The system provides three main modes of operation:

### 1. Train Model
Train the CNN model on FER2013 dataset:
```bash
python main.py train
```

### 2. Real-time Emotion Detection
Run basic real-time emotion detection:
```bash
python main.py detect
```

### 3. AR Emotion Overlay
Run AR-integrated emotion recognition with visual effects:
```bash
python main.py ar
```

## Project Structure

```
AR-Integrated-Real-Time-Emotion-Recognition/
├── src/
│   ├── data_loader.py          # FER2013 dataset loading and preprocessing
│   ├── emotion_model.py        # CNN model architecture
│   ├── real_time_detector.py   # Real-time emotion detection
│   └── ar_emotion_overlay.py   # AR overlay implementation
├── data/                       # Dataset directory (fer2013.csv)
├── models/                     # Saved model files
├── main.py                     # Command-line interface
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Model Architecture

The CNN model features:
- Multiple convolutional blocks with batch normalization
- MaxPooling and dropout layers for regularization
- Dense layers for classification
- Optimized for 48x48 grayscale face images
- 7-class emotion classification output

## Requirements

- Python 3.7+
- TensorFlow 2.13.0
- OpenCV 4.8.1
- MediaPipe 0.10.7
- NumPy, Pandas, Matplotlib
- Webcam for real-time detection

## Performance

The model achieves competitive accuracy on the FER2013 dataset and provides real-time emotion detection at interactive frame rates.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FER2013 dataset creators and Kaggle community
- MediaPipe for face detection capabilities
- TensorFlow team for deep learning framework
