"""
AR Overlay System
Handles rendering of emoji overlays and AR visualizations
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from config import (
    EMOTION_EMOJIS, 
    EMOJI_SIZE, 
    EMOJI_OFFSET_Y, 
    EMOJI_OFFSET_X,
    OVERLAY_ALPHA,
    FACE_BOX_COLOR,
    TEXT_COLOR,
    CONFIDENCE_BAR_COLOR,
    FONT_SCALE,
    FONT_THICKNESS
)

class AROverlay:
    def __init__(self):
        self.emoji_cache = {}
        self.default_font_size = 40
        
        # Try to load a better font for emojis
        try:
            # This might not work on all systems, so we have a fallback
            self.font = ImageFont.truetype("arial.ttf", self.default_font_size)
        except:
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.default_font_size)
            except:
                self.font = ImageFont.load_default()
    
    def create_emoji_image(self, emoji, size=EMOJI_SIZE):
        """Create an image with emoji text"""
        # Create a transparent image
        img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Calculate text position to center it
        bbox = draw.textbbox((0, 0), emoji, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size - text_width) // 2
        y = (size - text_height) // 2
        
        # Draw emoji
        draw.text((x, y), emoji, font=self.font, fill=(255, 255, 255, 255))
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
        
        return opencv_image
    
    def get_emoji_for_emotion(self, emotion):
        """Get emoji character for given emotion"""
        return EMOTION_EMOJIS.get(emotion, '😐')
    
    def overlay_emoji(self, frame, emotion, face_center, confidence=1.0):
        """Overlay emoji above detected face"""
        emoji_char = self.get_emoji_for_emotion(emotion)
        
        # Calculate emoji position
        center_x, center_y = face_center
        emoji_x = center_x + EMOJI_OFFSET_X - EMOJI_SIZE // 2
        emoji_y = center_y + EMOJI_OFFSET_Y - EMOJI_SIZE // 2
        
        # Ensure emoji stays within frame bounds
        emoji_x = max(0, min(emoji_x, frame.shape[1] - EMOJI_SIZE))
        emoji_y = max(0, min(emoji_y, frame.shape[0] - EMOJI_SIZE))
        
        # Create emoji image if not cached
        if emoji_char not in self.emoji_cache:
            self.emoji_cache[emoji_char] = self.create_emoji_image(emoji_char)
        
        emoji_img = self.emoji_cache[emoji_char].copy()
        
        # Adjust opacity based on confidence
        alpha = min(1.0, confidence * OVERLAY_ALPHA)
        
        # Apply emoji overlay
        self._apply_overlay(frame, emoji_img, emoji_x, emoji_y, alpha)
        
        return frame
    
    def _apply_overlay(self, background, overlay, x, y, alpha=1.0):
        """Apply overlay image onto background with transparency"""
        # Get overlay dimensions
        h, w = overlay.shape[:2]
        
        # Ensure overlay fits within background
        if x + w > background.shape[1]:
            w = background.shape[1] - x
            overlay = overlay[:, :w]
        if y + h > background.shape[0]:
            h = background.shape[0] - y
            overlay = overlay[:h, :]
        
        if w <= 0 or h <= 0:
            return
        
        # Get the region of interest
        roi = background[y:y+h, x:x+w]
        
        # Handle transparency
        if overlay.shape[2] == 4:  # BGRA
            # Separate alpha channel
            overlay_bgr = overlay[:, :, :3]
            overlay_alpha = overlay[:, :, 3] / 255.0 * alpha
            
            # Apply alpha blending
            for c in range(3):
                roi[:, :, c] = (1 - overlay_alpha) * roi[:, :, c] + \
                              overlay_alpha * overlay_bgr[:, :, c]
        else:  # BGR
            # Simple alpha blending
            cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
    
    def draw_face_info(self, frame, face_bbox, emotion, confidence):
        """Draw face bounding box and emotion information"""
        x, y, w, h = face_bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), FACE_BOX_COLOR, 2)
        
        # Prepare text
        emotion_text = f"{emotion}: {confidence:.2f}"
        
        # Calculate text size and position
        text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = x
        text_y = y - 10 if y - 10 > 0 else y + h + 20
        
        # Draw text background
        cv2.rectangle(frame, 
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0], text_y + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, emotion_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        return frame
    
    def draw_confidence_bar(self, frame, face_bbox, confidence):
        """Draw confidence bar near the face"""
        x, y, w, h = face_bbox
        
        # Bar dimensions
        bar_width = w
        bar_height = 6
        bar_x = x
        bar_y = y + h + 5
        
        # Ensure bar stays within frame
        if bar_y + bar_height > frame.shape[0]:
            bar_y = y - bar_height - 5
        
        # Draw background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Draw confidence bar
        confidence_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), 
                     CONFIDENCE_BAR_COLOR, -1)
        
        return frame
    
    def create_emotion_dashboard(self, frame, emotion_probabilities, emotion_labels):
        """Create a dashboard showing all emotion probabilities"""
        dashboard_height = 200
        dashboard_width = 300
        
        # Create dashboard background
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        dashboard.fill(50)  # Dark gray background
        
        # Draw emotion bars
        bar_height = 20
        bar_margin = 5
        start_y = 20
        
        for i, (emotion, prob) in enumerate(zip(emotion_labels, emotion_probabilities)):
            y = start_y + i * (bar_height + bar_margin)
            
            # Draw emotion label
            cv2.putText(dashboard, emotion[:8], (10, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
            
            # Draw probability bar
            bar_width = int((dashboard_width - 100) * prob)
            cv2.rectangle(dashboard, (90, y), (90 + bar_width, y + bar_height),
                         CONFIDENCE_BAR_COLOR, -1)
            
            # Draw probability value
            cv2.putText(dashboard, f"{prob:.2f}", (dashboard_width - 50, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_COLOR, 1)
        
        # Overlay dashboard on frame
        overlay_x = frame.shape[1] - dashboard_width - 10
        overlay_y = 10
        
        if overlay_x >= 0 and overlay_y >= 0:
            roi = frame[overlay_y:overlay_y+dashboard_height, overlay_x:overlay_x+dashboard_width]
            cv2.addWeighted(dashboard, 0.8, roi, 0.2, 0, roi)
        
        return frame
    
    def add_fps_counter(self, frame, fps):
        """Add FPS counter to frame"""
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame