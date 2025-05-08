import cv2
import numpy as np
import mediapipe as mp
import logging

# Configure logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('distraction_utils')

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define constants for face landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP = 1


def draw_landmarks(frame, face_landmarks, draw_mesh=False):
    """
    Draw facial landmarks on the frame
    
    Args:
        frame: Video frame
        face_landmarks: MediaPipe face landmarks
        draw_mesh: Whether to draw full face mesh
    """
    h, w, _ = frame.shape
    
    if draw_mesh:
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
    
    # Draw eyes
    for idx in LEFT_EYE + RIGHT_EYE:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # Draw nose tip
    nose = face_landmarks.landmark[NOSE_TIP]
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)
    
    # Draw face direction line
    center_x = w // 2
    cv2.line(frame, (center_x, 0), (center_x, h), (255, 0, 0), 1)
    cv2.line(frame, (nose_x, nose_y), (center_x, nose_y), (0, 255, 255), 2)
    
    return frame


def calculate_ear(landmarks, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio (EAR) from face landmarks
    
    Args:
        landmarks: MediaPipe face landmarks
        frame_width: Width of the video frame
        frame_height: Height of the video frame
    
    Returns:
        float: Average EAR value for both eyes
    """
    try:
        left_eye = np.array([[landmarks[i].x * frame_width, landmarks[i].y * frame_height] for i in LEFT_EYE])
        right_eye = np.array([[landmarks[i].x * frame_width, landmarks[i].y * frame_height] for i in RIGHT_EYE])
        
        # Calculate distances for EAR
        left_A = np.linalg.norm(left_eye[1] - left_eye[5])
        left_B = np.linalg.norm(left_eye[2] - left_eye[4])
        left_C = np.linalg.norm(left_eye[0] - left_eye[3])
        
        right_A = np.linalg.norm(right_eye[1] - right_eye[5])
        right_B = np.linalg.norm(right_eye[2] - right_eye[4])
        right_C = np.linalg.norm(right_eye[0] - right_eye[3])
        
        # Calculate EAR for each eye
        left_ear = (left_A + left_B) / (2.0 * left_C)
        right_ear = (right_A + right_B) / (2.0 * right_C)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        return avg_ear
    except Exception as e:
        logger.error(f"Error calculating EAR: {e}")
        return 0.3  # Default value if calculation fails


def get_face_direction(landmarks, frame_width):
    """
    Calculate face direction based on nose position
    
    Args:
        landmarks: MediaPipe face landmarks
        frame_width: Width of the video frame
    
    Returns:
        float: Deviation from center (positive = right, negative = left)
    """
    nose_x = landmarks[NOSE_TIP].x * frame_width
    center_x = frame_width / 2
    return nose_x - center_x