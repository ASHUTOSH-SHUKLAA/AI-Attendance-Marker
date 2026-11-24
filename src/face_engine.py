"""
Face Recognition Engine.
Handles face detection, embedding extraction, and metric matching.
Supports both the standard `face_recognition` (dlib) library and OpenCV fallbacks.
"""
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2

import config

# Try importing face_recognition (primary assignment recommendation)
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False


class DetectedFace:
    """Represents a face detected in a frame with identity and confidence."""

    def __init__(
        self,
        box: Tuple[int, int, int, int],  # (top, right, bottom, left)
        student_id: Optional[str] = None,
        name: str = "Unknown",
        confidence: float = 0.0,
        distance: float = 1.0,
        encoding: Optional[np.ndarray] = None,
    ):
        self.top, self.right, self.bottom, self.left = box
        self.student_id = student_id
        self.name = name
        self.confidence = confidence
        self.distance = distance
        self.encoding = encoding

    @property
    def is_known(self) -> bool:
        return self.name != "Unknown" and self.student_id is not None

    def as_rect(self) -> Tuple[int, int, int, int]:
        """Returns (x, y, w, h) format for OpenCV drawing."""
        return (self.left, self.top, self.right - self.left, self.bottom - self.top)


class FaceEngine:
    """
    Core Computer Vision engine for classroom face detection,
    feature extraction, and similarity matching.
    """

    def __init__(
        self,
        encodings_path: Optional[Path] = None,
        tolerance: float = config.FACE_DISTANCE_TOLERANCE,
        detection_model: str = config.DETECTION_MODEL,
    ):
        self.encodings_path = encodings_path or config.ENCODINGS_FILE
        self.tolerance = tolerance
        self.detection_model = detection_model

        # In-memory database of registered faces
        self.known_encodings: List[np.ndarray] = []
        self.known_ids: List[str] = []
        self.known_names: List[str] = []

        # Cached face cascade detector
        self.face_cascade = self._load_cascade()

        # Check and purge any legacy mock data automatically
        if self.encodings_path.exists() and self.encodings_path.stat().st_size > 0:
            try:
                with open(self.encodings_path, "rb") as f:
                    _check_data = pickle.load(f)
                mock_names = {
                    'Aarav Sharma', 'Ananya Verma', 'Rohan Patel', 'Priya Singh',
                    'Ishaan Gupta', 'Sneha Reddy', 'Vikram Malhotra', 'Neha Joshi',
                    'Arjun Nair', 'Tanvi Kapoor'
                }
                if any(name in mock_names for name in _check_data.get("names", [])):
                    from src.database import reset_all_data
                    reset_all_data()
            except Exception:
                pass

        # Load encodings if available
        self.load_encodings()

    @property
    def total_registered(self) -> int:
        return len(self.known_encodings)

    def load_encodings(self) -> bool:
        """Loads serialized face encodings from disk."""
        if not self.encodings_path.exists() or self.encodings_path.stat().st_size == 0:
            print(f"[FaceEngine] No pre-saved encodings found at {self.encodings_path}")
            self.known_encodings = []
            self.known_ids = []
            self.known_names = []
            return False

        try:
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
                raw_encodings = data.get("encodings", [])
                raw_ids = data.get("student_ids", [])
                raw_names = data.get("names", [])

            # Filter out any mock/synthetic students automatically
            mock_names = {
                'Aarav Sharma', 'Ananya Verma', 'Rohan Patel', 'Priya Singh',
                'Ishaan Gupta', 'Sneha Reddy', 'Vikram Malhotra', 'Neha Joshi',
                'Arjun Nair', 'Tanvi Kapoor'
            }
            clean_encs, clean_ids, clean_names = [], [], []
            has_mock = False
            for enc, sid, sname in zip(raw_encodings, raw_ids, raw_names):
                if sname in mock_names:
                    has_mock = True
                else:
                    clean_encs.append(enc)
                    clean_ids.append(sid)
                    clean_names.append(sname)

            self.known_encodings = clean_encs
            self.known_ids = clean_ids
            self.known_names = clean_names

            if has_mock:
                print(f"[FaceEngine] Purged mock synthetic student data. Enrolled real students: {len(clean_ids)}")
                self.save_encodings()

            print(
                f"[FaceEngine] Loaded {len(self.known_encodings)} face encodings for "
                f"{len(set(self.known_ids))} real enrolled students."
            )
            return True
        except Exception as e:
            print(f"[FaceEngine Error] Failed to load encodings: {e}")
            return False

    def save_encodings(self) -> bool:
        """Saves current known encodings to disk."""
        try:
            self.encodings_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "encodings": self.known_encodings,
                "student_ids": self.known_ids,
                "names": self.known_names,
            }
            with open(self.encodings_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[FaceEngine] Successfully saved {len(self.known_encodings)} encodings to {self.encodings_path}")
            return True
        except Exception as e:
            print(f"[FaceEngine Error] Failed to save encodings: {e}")
            return False

    def register_face(
        self,
        student_id: str,
        name: str,
        encoding: np.ndarray,
        auto_save: bool = True,
    ) -> None:
        """Appends a new face vector to the in-memory database."""
        self.known_encodings.append(encoding)
        self.known_ids.append(student_id)
        self.known_names.append(name)
        if auto_save:
            self.save_encodings()

    def extract_face_encodings_from_image(
        self,
        bgr_image: np.ndarray,
        num_jitters: int = config.NUM_JITTERS,
    ) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Detects faces and computes 128D embeddings for all faces in an image.
        Returns list of tuples: ( (top, right, bottom, left), encoding_vector )
        """
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if HAS_FACE_RECOGNITION:
            # 1. Detection
            face_locations = face_recognition.face_locations(
                rgb_image,
                model=self.detection_model,
            )
            # 2. Embedding extraction
            encodings = face_recognition.face_encodings(
                rgb_image,
                known_face_locations=face_locations,
                num_jitters=num_jitters,
            )
            return list(zip(face_locations, encodings))
        else:
            # Fallback when dlib/face_recognition is not yet installed:
            # Uses OpenCV Haar Cascade + normalized grayscale feature representation
            return self._fallback_extract_faces(bgr_image)

    def _load_cascade(self) -> cv2.CascadeClassifier:
        """Loads and caches the Haar Cascade classifier from local models or OpenCV data."""
        # 1. Project models directory
        local_path = config.MODELS_DIR / "haarcascade_frontalface_default.xml"
        if local_path.exists() and local_path.stat().st_size > 1000:
            c = cv2.CascadeClassifier(str(local_path))
            if not c.empty():
                return c

        # 2. OpenCV data directory
        try:
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                p = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
                if p.exists():
                    c = cv2.CascadeClassifier(str(p))
                    if not c.empty():
                        return c
        except Exception:
            pass

        # 3. Auto-download into models/
        try:
            from download_models import download_cascade
            downloaded = download_cascade()
            if downloaded.exists() and downloaded.stat().st_size > 1000:
                c = cv2.CascadeClassifier(str(downloaded))
                if not c.empty():
                    return c
        except Exception as e:
            print(f"[FaceEngine Warning] Cascade resolution fallback: {e}")

        return cv2.CascadeClassifier()

    def _fallback_extract_faces(
        self,
        bgr_image: np.ndarray,
    ) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        OpenCV fallback feature extractor for environments without dlib installed.
        Uses Haar Cascade for detection and normalized 128-bin gradient/spatial features.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        if self.face_cascade is None or self.face_cascade.empty():
            self.face_cascade = self._load_cascade()

        faces = []
        if not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

        results = []
        for (x, y, w, h) in faces:
            top, right, bottom, left = y, x + w, y + h, x
            face_roi = gray[y:y + h, x:x + w]
            # 1. Normalize illumination
            eq_roi = cv2.equalizeHist(face_roi)
            resized = cv2.resize(eq_roi, (96, 96))

            # 2. Extract 8x8 spatial grid features (64 cells x 2 = 128 dimensions)
            cell_means = []
            cell_stds = []
            for r in range(8):
                for c in range(8):
                    cell = resized[r * 12:(r + 1) * 12, c * 12:(c + 1) * 12]
                    cell_means.append(float(np.mean(cell)))
                    cell_stds.append(float(np.std(cell)))

            descriptor = np.array(cell_means + cell_stds, dtype=np.float32)
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
            results.append(((top, right, bottom, left), descriptor))
        return results

    def match_encoding(self, candidate_encoding: np.ndarray) -> Tuple[Optional[str], str, float, float]:
        """
        Compares candidate encoding against all registered encodings.
        Returns (student_id, name, confidence_pct, best_distance).
        """
        if len(self.known_encodings) == 0:
            return None, "Unknown", 0.0, 1.0

        if HAS_FACE_RECOGNITION:
            # Euclidean distance across 128D space
            distances = face_recognition.face_distance(self.known_encodings, candidate_encoding)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist <= self.tolerance:
                confidence = max(0.0, min(100.0, (1.0 - (best_dist / (self.tolerance * 1.5))) * 100.0))
                return self.known_ids[best_idx], self.known_names[best_idx], round(confidence, 2), round(best_dist, 3)
            else:
                return None, "Unknown", 0.0, round(best_dist, 3)
        else:
            # Cosine distance for normalized spatial cell descriptors
            distances = [
                float(1.0 - np.dot(candidate_encoding, known) / (np.linalg.norm(candidate_encoding) * np.linalg.norm(known) + 1e-7))
                for known in self.known_encodings
            ]
            best_idx = int(np.argmin(distances))
            best_dist = distances[best_idx]

            # Strict threshold: requires close spatial alignment to prevent false matches
            strict_threshold = 0.12
            if best_dist <= strict_threshold:
                confidence = max(0.0, min(100.0, (1.0 - (best_dist / (strict_threshold * 1.5))) * 100.0))
                return self.known_ids[best_idx], self.known_names[best_idx], round(confidence, 2), round(best_dist, 3)
            return None, "Unknown", 0.0, round(best_dist, 3)

    def process_frame(
        self,
        frame: np.ndarray,
        scale: float = 0.5,
    ) -> List[DetectedFace]:
        """
        High-throughput multi-face recognition for a single video frame.
        Scales frame down for fast detection, then scales bounding boxes back up.
        """
        # Downscale for performance
        if scale != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = frame

        faces_data = self.extract_face_encodings_from_image(small_frame)
        detected_faces = []

        inv_scale = 1.0 / scale
        for (top, right, bottom, left), encoding in faces_data:
            # Scale coordinates back to original frame size
            orig_top = int(top * inv_scale)
            orig_right = int(right * inv_scale)
            orig_bottom = int(bottom * inv_scale)
            orig_left = int(left * inv_scale)

            student_id, name, confidence, distance = self.match_encoding(encoding)

            detected_faces.append(
                DetectedFace(
                    box=(orig_top, orig_right, orig_bottom, orig_left),
                    student_id=student_id,
                    name=name,
                    confidence=confidence,
                    distance=distance,
                    encoding=encoding,
                )
            )

        return detected_faces
