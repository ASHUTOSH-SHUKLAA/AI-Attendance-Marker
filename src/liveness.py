"""
Anti-Spoofing and Liveness Detection Module.
Prevents attendance fraud from printed photographs or smartphone screen replays.
Implements:
1. Eye Aspect Ratio (EAR) blink detection via facial landmarks.
2. Laplacian Variance Texture Analysis (detects flat/moire artifacts from screen re-captures).
"""
import numpy as np
import cv2
from typing import Tuple, Optional, List

try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False


class LivenessDetector:
    """
    Evaluates face crops or video sequences to ensure the subject is physically present.
    """

    # Typical EAR threshold for a blink
    EAR_THRESHOLD = 0.22
    # Texture blur threshold for printed photo detection
    BLUR_THRESHOLD = 80.0

    @staticmethod
    def calculate_ear(eye_landmarks: List[Tuple[int, int]]) -> float:
        """
        Computes the Eye Aspect Ratio (EAR) given 6 (x, y) landmark points for an eye.
        Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        if len(eye_landmarks) < 6:
            return 0.30

        p1, p2, p3, p4, p5, p6 = [np.array(pt) for pt in eye_landmarks[:6]]

        # Vertical distances
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)

        # Horizontal distance
        horizontal = np.linalg.norm(p1 - p4)

        if horizontal == 0:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(ear)

    @classmethod
    def check_texture_liveness(cls, face_bgr_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Uses Laplacian variance to detect low-frequency blur or screen capture grain.
        Returns (is_live, variance_score).
        """
        if face_bgr_crop is None or face_bgr_crop.size == 0:
            return False, 0.0

        gray = cv2.cvtColor(face_bgr_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_live = laplacian_var >= cls.BLUR_THRESHOLD
        return is_live, round(laplacian_var, 2)

    @classmethod
    def check_eye_blink(cls, rgb_image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        Computes average EAR across both eyes using 68-point facial landmarks.
        Returns (is_blinking, avg_ear).
        """
        if not HAS_FACE_RECOGNITION:
            # When dlib is unavailable, return neutral placeholder
            return True, 0.30

        landmarks_list = face_recognition.face_landmarks(rgb_image, face_locations=[face_location])
        if not landmarks_list:
            return True, 0.30

        landmarks = landmarks_list[0]
        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])

        if not left_eye or not right_eye:
            return True, 0.30

        left_ear = cls.calculate_ear(left_eye)
        right_ear = cls.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        is_blinking = avg_ear < cls.EAR_THRESHOLD
        return is_blinking, round(avg_ear, 3)
