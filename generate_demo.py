#!/usr/bin/env python3
"""
Visual Demo Generator for AR Emotion Recognition System
Creates demonstration images showing the system in action
"""

import cv2
import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from ar_overlay import AROverlay
from config import EMOTION_LABELS, EMOTION_EMOJIS

def create_demo_image():
    """Create a demonstration image showing the AR emotion recognition system"""
    
    # Create base image
    width, height = 800, 600
    demo_image = np.zeros((height, width, 3), dtype=np.uint8)
    demo_image.fill(40)  # Dark background
    
    # Initialize AR overlay
    ar_overlay = AROverlay()
    
    # Title
    cv2.putText(demo_image, "AR EMOTION RECOGNITION SYSTEM", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.putText(demo_image, "Real-time facial emotion detection with AR emoji overlays", 
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Create simulated faces with different emotions
    face_positions = [
        (150, 150, 100, 120, "Happy"),    # x, y, w, h, emotion
        (350, 150, 100, 120, "Sad"), 
        (550, 150, 100, 120, "Surprise"),
        (150, 350, 100, 120, "Angry"),
        (350, 350, 100, 120, "Fear"),
        (550, 350, 100, 120, "Neutral")
    ]
    
    for x, y, w, h, emotion in face_positions:
        # Draw face rectangle
        cv2.rectangle(demo_image, (x, y), (x + w, y + h), (100, 100, 100), -1)
        
        # Draw simple face features
        # Eyes
        eye_y = y + h // 3
        cv2.circle(demo_image, (x + w // 3, eye_y), 8, (255, 255, 255), -1)
        cv2.circle(demo_image, (x + 2 * w // 3, eye_y), 8, (255, 255, 255), -1)
        
        # Mouth (different for each emotion)
        mouth_y = y + 2 * h // 3
        mouth_center = (x + w // 2, mouth_y)
        
        if emotion == "Happy":
            # Smile
            cv2.ellipse(demo_image, mouth_center, (20, 10), 0, 0, 180, (255, 255, 255), 2)
        elif emotion == "Sad":
            # Frown
            cv2.ellipse(demo_image, mouth_center, (20, 10), 0, 180, 360, (255, 255, 255), 2)
        else:
            # Neutral mouth
            cv2.ellipse(demo_image, mouth_center, (15, 5), 0, 0, 180, (255, 255, 255), 2)
        
        # Draw face bounding box
        cv2.rectangle(demo_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add emotion label
        cv2.putText(demo_image, f"{emotion}: 0.{85 + np.random.randint(0, 14)}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add AR emoji overlay
        face_center = (x + w // 2, y + h // 2)
        demo_image = ar_overlay.overlay_emoji(demo_image, emotion, face_center, 0.9)
        
        # Add confidence bar
        bar_y = y + h + 10
        bar_width = w
        confidence = 0.75 + np.random.random() * 0.2
        
        # Background bar
        cv2.rectangle(demo_image, (x, bar_y), (x + bar_width, bar_y + 6), (50, 50, 50), -1)
        
        # Confidence bar
        conf_width = int(bar_width * confidence)
        cv2.rectangle(demo_image, (x, bar_y), (x + conf_width, bar_y + 6), (0, 255, 255), -1)
    
    # Add feature list
    features = [
        "✓ Real-time face detection",
        "✓ 7 emotion categories", 
        "✓ AR emoji overlays",
        "✓ Confidence visualization",
        "✓ Multi-face support",
        "✓ Performance optimized"
    ]
    
    start_y = 120
    for i, feature in enumerate(features):
        cv2.putText(demo_image, feature, (50, start_y + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Add system info
    cv2.putText(demo_image, "System Components:", (50, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    components = [
        "• CNN Emotion Recognition",
        "• OpenCV/MediaPipe Face Detection", 
        "• AR Overlay System",
        "• Performance Monitor"
    ]
    
    for i, component in enumerate(components):
        cv2.putText(demo_image, component, (50, 350 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Add FPS counter
    cv2.putText(demo_image, "FPS: 30.0", (10, height - 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add face count
    cv2.putText(demo_image, f"Faces: {len(face_positions)}", (10, height - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add controls info
    cv2.putText(demo_image, "Controls: Q=Quit, D=Debug, C=Confidence, E=Dashboard", 
               (width - 500, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return demo_image

def create_architecture_diagram():
    """Create a system architecture diagram"""
    
    width, height = 800, 600
    arch_image = np.zeros((height, width, 3), dtype=np.uint8)
    arch_image.fill(30)
    
    # Title
    cv2.putText(arch_image, "SYSTEM ARCHITECTURE", 
                (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Define components with positions
    components = [
        ("Camera Input", (100, 100), (150, 60)),
        ("Face Detection", (300, 100), (150, 60)),
        ("Emotion CNN", (500, 100), (150, 60)),
        ("AR Overlay", (300, 220), (150, 60)),
        ("Display Output", (500, 220), (150, 60)),
        ("Performance\nMonitor", (100, 220), (150, 60)),
        ("Analytics", (100, 340), (150, 60)),
        ("Configuration", (300, 340), (150, 60))
    ]
    
    # Draw components
    for name, (x, y), (w, h) in components:
        # Component box
        cv2.rectangle(arch_image, (x, y), (x + w, y + h), (0, 100, 200), -1)
        cv2.rectangle(arch_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Component text
        lines = name.split('\n')
        for i, line in enumerate(lines):
            text_y = y + h // 2 + (i - len(lines)//2) * 15
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (w - text_size[0]) // 2
            cv2.putText(arch_image, line, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw arrows
    arrows = [
        ((175, 130), (300, 130)),  # Camera -> Face Detection
        ((450, 130), (500, 130)),  # Face Detection -> Emotion CNN
        ((375, 160), (375, 220)),  # Face Detection -> AR Overlay
        ((450, 250), (500, 250)),  # AR Overlay -> Display
        ((175, 160), (175, 220)),  # Camera -> Performance Monitor
        ((175, 280), (175, 340)),  # Performance -> Analytics
        ((250, 370), (300, 370)),  # Analytics -> Configuration
    ]
    
    for (start_x, start_y), (end_x, end_y) in arrows:
        cv2.arrowedLine(arch_image, (start_x, start_y), (end_x, end_y), 
                       (0, 255, 255), 2, tipLength=0.3)
    
    # Add data flow labels
    cv2.putText(arch_image, "Video Stream", (180, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(arch_image, "Face ROI", (460, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(arch_image, "Emotions", (380, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(arch_image, "AR Frame", (460, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Add technology stack
    cv2.putText(arch_image, "Technology Stack:", (50, 480), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    tech_stack = [
        "• OpenCV - Computer Vision",
        "• TensorFlow - Deep Learning",
        "• MediaPipe - Face Detection", 
        "• NumPy - Numerical Computing",
        "• PIL - Image Processing"
    ]
    
    for i, tech in enumerate(tech_stack):
        cv2.putText(arch_image, tech, (50, 510 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return arch_image

def main():
    """Generate demonstration images"""
    print("Generating AR Emotion Recognition Demo Images...")
    
    try:
        # Create demo image
        demo_img = create_demo_image()
        cv2.imwrite("demo_screenshot.png", demo_img)
        print("✓ Demo screenshot saved as 'demo_screenshot.png'")
        
        # Create architecture diagram
        arch_img = create_architecture_diagram()
        cv2.imwrite("architecture_diagram.png", arch_img)
        print("✓ Architecture diagram saved as 'architecture_diagram.png'")
        
        print("\nDemo images created successfully!")
        print("Files created:")
        print("- demo_screenshot.png: Shows the system in action")
        print("- architecture_diagram.png: Shows system architecture")
        
        return True
        
    except Exception as e:
        print(f"Error generating demo images: {e}")
        return False

if __name__ == "__main__":
    main()