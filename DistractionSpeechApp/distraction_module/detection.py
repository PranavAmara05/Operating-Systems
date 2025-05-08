import cv2
import time
import threading
import logging
import mediapipe as mp
import numpy as np
import pyttsx3
from collections import deque
from .utils import calculate_ear, draw_landmarks, get_face_direction

# Configure logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('distraction_detector')

class DistractionDetector:
    """
    Detects user distraction by analyzing face position and eye closure
    """
    
    def __init__(self, semaphore=None):
        """Initialize the distraction detector
        
        Args:
            semaphore: Optional threading.Semaphore for synchronization
        """
        self.distraction_count = 0
        self.semaphore = semaphore
        
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        
        # MediaPipe configuration
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True, 
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Detection thresholds
        self.FACE_TURN_THRESHOLD = 2.0  # seconds
        self.EYE_CLOSED_THRESHOLD = 4.0  # seconds
        self.EAR_THRESHOLD = 0.25  # eye aspect ratio threshold
        self.FACE_DEVIATION_THRESHOLD = 80  # pixels from center
        self.FRAME_SMOOTHING_WINDOW = 5  # frames for smoothing
        
        # Buffers for smoothing measurements
        self.ear_buffer = deque(maxlen=self.FRAME_SMOOTHING_WINDOW)
        self.face_dir_buffer = deque(maxlen=self.FRAME_SMOOTHING_WINDOW)
        
        # State variables
        self.face_turn_start = None
        self.eye_close_start = None
        self.in_distraction = False
        self.last_distraction_time = 0
        self.running = False
        self.video_feed = None
        
        logger.info("Distraction detector initialized")
    
    def speak(self, message):
        """Use text-to-speech to alert the user
        
        Args:
            message: Text to speak
        """
        try:
            logger.info(f"Speaking alert: {message}")
            threading.Thread(target=self._speak_thread, args=(message,), daemon=True).start()
        except Exception as e:
            logger.error(f"Error in speak function: {e}")
    
    def _speak_thread(self, message):
        """Thread function for non-blocking speech
        
        Args:
            message: Text to speak
        """
        try:
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in speak thread: {e}")
    
    def start(self, camera_index=0):
        """Start distraction detection on a background thread
        
        Args:
            camera_index: Camera device index
        """
        if self.running:
            logger.warning("Distraction detector already running")
            return
        
        self.video_feed = cv2.VideoCapture(camera_index)
        if not self.video_feed.isOpened():
            logger.error(f"Could not open camera at index {camera_index}")
            return
        
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        logger.info("Distraction detection started")
    
    def stop(self):
        """Stop the distraction detection thread"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        
        if self.video_feed:
            self.video_feed.release()
        
        logger.info("Distraction detection stopped")
    
    def _detection_loop(self):
        """Main detection loop that processes video frames"""
        if self.semaphore:
            self.semaphore.acquire()
            logger.info("Acquired semaphore for distraction detection")
        
        try:
            while self.running and self.video_feed.isOpened():
                ret, frame = self.video_feed.read()
                if not ret:
                    logger.warning("Failed to get frame from camera")
                    break
                
                # Process the frame
                self._process_frame(frame)
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
        finally:
            if self.semaphore:
                self.semaphore.release()
                logger.info("Released semaphore from distraction detection")
    
    def _process_frame(self, frame):
        """Process a single video frame to detect distraction
        
        Args:
            frame: Video frame from camera
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        now = time.time()
        distraction_detected = False
        distraction_type = None
        
        # Check if face is detected
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = face.landmark
            
            # Calculate face direction
            face_deviation = abs(get_face_direction(landmarks, w))
            self.face_dir_buffer.append(face_deviation)
            avg_deviation = sum(self.face_dir_buffer) / len(self.face_dir_buffer)
            
            # Check for face turned away
            if avg_deviation > self.FACE_DEVIATION_THRESHOLD:
                if self.face_turn_start is None:
                    self.face_turn_start = now
                elif now - self.face_turn_start > self.FACE_TURN_THRESHOLD:
                    distraction_detected = True
                    distraction_type = "Face turned"
            else:
                self.face_turn_start = None
            
            # Calculate eye aspect ratio
            ear = calculate_ear(landmarks, w, h)
            self.ear_buffer.append(ear)
            avg_ear = sum(self.ear_buffer) / len(self.ear_buffer)
            
            # Check for eyes closed
            if avg_ear < self.EAR_THRESHOLD:
                if self.eye_close_start is None:
                    self.eye_close_start = now
                elif now - self.eye_close_start > self.EYE_CLOSED_THRESHOLD:
                    distraction_detected = True
                    distraction_type = "Eyes closed"
            else:
                self.eye_close_start = None
        else:
            # No face detected
            distraction_detected = True
            distraction_type = "No face"
        
        # Handle distraction state changes
        if distraction_detected and not self.in_distraction:
            self.distraction_count += 1
            self.in_distraction = True
            self.last_distraction_time = now
            
            # Alert based on distraction type
            if distraction_type == "Eyes closed":
                self.speak("Don't sleep now!")
            elif distraction_type == "Face turned":
                self.speak("Please stay focused")
            elif distraction_type == "No face":
                self.speak("Please return to your seat")
            
            logger.info(f"Distraction detected: {distraction_type} - Count: {self.distraction_count}")
        
        # Reset distraction state after delay
        if not distraction_detected and self.in_distraction:
            if (now - self.last_distraction_time) > 2:
                self.in_distraction = False
                logger.info("User returned to focused state")
    
    def get_distraction_count(self):
        """Get the total number of distractions detected
        
        Returns:
            int: Number of distractions
        """
        return self.distraction_count