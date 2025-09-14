"""
Camera and Video Processing System
Handles camera input, frame processing, and performance optimization
"""

import cv2
import numpy as np
import time
from threading import Thread, Lock
from queue import Queue
from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT, 
    CAMERA_FPS,
    SKIP_FRAMES
)

class CameraProcessor:
    def __init__(self, source=0, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.is_recording = False
        
        # Threading for performance
        self.frame_queue = Queue(maxsize=5)
        self.processing_thread = None
        self.frame_lock = Lock()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Frame skipping for optimization
        self.frame_skip_counter = 0
        self.skip_frames = SKIP_FRAMES
        
    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def start_capture(self):
        """Start camera capture"""
        if not self.initialize_camera():
            return False
        
        self.is_recording = True
        return True
    
    def stop_capture(self):
        """Stop camera capture"""
        self.is_recording = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def read_frame(self):
        """Read a frame from camera"""
        if not self.cap or not self.is_recording:
            return None, False
        
        ret, frame = self.cap.read()
        
        if ret:
            # Update FPS counter
            self._update_fps()
            
            # Resize frame if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
        
        return frame, ret
    
    def should_process_frame(self):
        """Determine if current frame should be processed (for optimization)"""
        self.frame_skip_counter += 1
        
        if self.frame_skip_counter >= self.skip_frames:
            self.frame_skip_counter = 0
            return True
        
        return False
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_fps(self):
        """Get current FPS"""
        return self.current_fps
    
    def flip_frame(self, frame, horizontal=True):
        """Flip frame horizontally or vertically"""
        if horizontal:
            return cv2.flip(frame, 1)  # Horizontal flip
        else:
            return cv2.flip(frame, 0)  # Vertical flip
    
    def resize_frame(self, frame, target_size):
        """Resize frame to target size"""
        return cv2.resize(frame, target_size)
    
    def apply_filters(self, frame, filters=None):
        """Apply various filters to improve face detection"""
        if filters is None:
            filters = []
        
        processed_frame = frame.copy()
        
        for filter_name in filters:
            if filter_name == 'brightness':
                processed_frame = self._adjust_brightness(processed_frame, 1.2)
            elif filter_name == 'contrast':
                processed_frame = self._adjust_contrast(processed_frame, 1.1)
            elif filter_name == 'blur_reduction':
                processed_frame = self._reduce_blur(processed_frame)
            elif filter_name == 'noise_reduction':
                processed_frame = self._reduce_noise(processed_frame)
        
        return processed_frame
    
    def _adjust_brightness(self, frame, factor):
        """Adjust frame brightness"""
        return cv2.convertScaleAbs(frame, alpha=factor, beta=30)
    
    def _adjust_contrast(self, frame, factor):
        """Adjust frame contrast"""
        return cv2.convertScaleAbs(frame, alpha=factor, beta=0)
    
    def _reduce_blur(self, frame):
        """Apply sharpening filter to reduce blur"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(frame, -1, kernel)
    
    def _reduce_noise(self, frame):
        """Apply noise reduction filter"""
        return cv2.bilateralFilter(frame, 9, 75, 75)
    
    def get_frame_info(self, frame):
        """Get information about the frame"""
        if frame is None:
            return None
        
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'dtype': frame.dtype,
            'size': frame.size
        }


class VideoWriter:
    def __init__(self, output_path, width, height, fps=30):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        self.is_recording = False
    
    def start_recording(self):
        """Start video recording"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        if self.writer.isOpened():
            self.is_recording = True
            print(f"Started recording to {self.output_path}")
            return True
        else:
            print("Failed to start video recording")
            return False
    
    def write_frame(self, frame):
        """Write frame to video file"""
        if self.is_recording and self.writer:
            # Resize frame if necessary
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.writer.write(frame)
    
    def stop_recording(self):
        """Stop video recording"""
        if self.writer:
            self.writer.release()
            self.writer = None
            self.is_recording = False
            print(f"Recording saved to {self.output_path}")


class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.processing_times = []
        
    def start_frame_timing(self):
        """Start timing a frame processing"""
        return time.time()
    
    def end_frame_timing(self, start_time):
        """End timing a frame processing"""
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Keep only last 30 measurements
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        return processing_time
    
    def get_average_processing_time(self):
        """Get average processing time"""
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_performance_stats(self):
        """Get performance statistics"""
        runtime = time.time() - self.start_time
        avg_processing_time = self.get_average_processing_time()
        
        return {
            'runtime': runtime,
            'frames_processed': self.frame_count,
            'avg_fps': self.frame_count / runtime if runtime > 0 else 0,
            'avg_processing_time': avg_processing_time,
            'efficiency': (1 / avg_processing_time) if avg_processing_time > 0 else 0
        }
    
    def increment_frame_count(self):
        """Increment processed frame counter"""
        self.frame_count += 1