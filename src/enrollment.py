"""
Student Enrollment Module.
Allows registering new students through live webcam capture or batch image folders.
Computes and serializes 128D face embeddings into `data/encodings.pkl`.
"""
import os
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

import config
from src.database import DatabaseManager
from src.face_engine import FaceEngine


class EnrollmentManager:
    """Handles enrollment of student identities, image persistence, and feature extraction."""

    def __init__(self, db: Optional[DatabaseManager] = None, engine: Optional[FaceEngine] = None):
        self.db = db or DatabaseManager()
        self.engine = engine or FaceEngine()

    def enroll_from_images(
        self,
        student_id: str,
        name: str,
        roll_no: str,
        image_paths: List[Path],
        branch: str = "CSE",
        semester: int = 5,
    ) -> bool:
        """
        Enrolls a student from a list of image file paths.
        Extracts face embeddings and stores them in the vector database and SQLite.
        """
        encodings = []
        student_folder = config.STUDENTS_DIR / f"{student_id}_{name.replace(' ', '_')}"
        student_folder.mkdir(parents=True, exist_ok=True)

        for img_idx, img_path in enumerate(image_paths):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[Warning] Could not read image: {img_path}")
                continue

            extracted = self.engine.extract_face_encodings_from_image(bgr)
            if not extracted:
                print(f"[Warning] No face detected in: {img_path.name}")
                continue

            # Pick the most prominent/largest face if multiple detected
            _, encoding = extracted[0]
            encodings.append(encoding)

            # Copy or save image into student directory
            dest = student_folder / f"sample_{img_idx + 1:02d}.jpg"
            cv2.imwrite(str(dest), bgr)

        if not encodings:
            print(f"[Enrollment Error] Failed to extract any valid face encodings for {name} ({student_id}).")
            return False

        # Register all valid encodings in FaceEngine
        for enc in encodings:
            self.engine.register_face(student_id=student_id, name=name, encoding=enc, auto_save=False)

        self.engine.save_encodings()

        # Save student profile into SQLite
        self.db.add_student(
            student_id=student_id,
            name=name,
            roll_no=roll_no,
            branch=branch,
            semester=semester,
        )

        print(f"[Enrollment Success] Successfully registered {name} ({student_id}) with {len(encodings)} face samples.")
        return True

    def enroll_from_webcam(
        self,
        student_id: str,
        name: str,
        roll_no: str,
        num_samples: int = 5,
        camera_index: int = config.CAMERA_INDEX,
        branch: str = "CSE",
        semester: int = 5,
    ) -> bool:
        """
        Interactive webcam enrollment tool.
        Captures multiple facial angles (front, slight turn) with live on-screen countdown.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[Error] Cannot access camera at index {camera_index}.")
            return False

        student_folder = config.STUDENTS_DIR / f"{student_id}_{name.replace(' ', '_')}"
        student_folder.mkdir(parents=True, exist_ok=True)

        captured_images: List[np.ndarray] = []
        captured_encodings: List[np.ndarray] = []

        print("\n=======================================================")
        print(f" Starting live enrollment for: {name} (Roll: {roll_no})")
        print(f" Press [SPACE] to capture sample | [Q] to abort")
        print("=======================================================\n")

        sample_count = 0
        last_capture_time = 0

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            display = frame.copy()

            # Draw a centered framing box for the user's face
            box_w, box_h = int(w * 0.45), int(h * 0.6)
            top = int((h - box_h) / 2)
            left = int((w - box_w) / 2)
            cv2.rectangle(display, (left, top), (left + box_w, top + box_h), (0, 255, 0), 2)

            # Display header HUD
            hud_text = f"Enrolling: {name} ({student_id}) - Sample {sample_count + 1}/{num_samples}"
            cv2.putText(display, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Align face in box & Press SPACE to capture (Q to quit)", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("Student Enrollment", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[Enrollment] Cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                return False

            if key == 32:  # Spacebar
                current_time = time.time()
                if current_time - last_capture_time < 0.5:
                    continue  # debounce accidental double-clicks

                extracted = self.engine.extract_face_encodings_from_image(frame)
                if not extracted:
                    print("[Warning] No face detected. Please ensure good lighting and look into the camera.")
                    continue

                _, encoding = extracted[0]
                captured_encodings.append(encoding)
                captured_images.append(frame.copy())
                sample_count += 1
                last_capture_time = current_time

                # Save frame to disk
                img_path = student_folder / f"sample_{sample_count:02d}.jpg"
                cv2.imwrite(str(img_path), frame)
                print(f" -> Captured sample {sample_count}/{num_samples}")

        cap.release()
        cv2.destroyAllWindows()

        # Register embeddings
        for enc in captured_encodings:
            self.engine.register_face(student_id=student_id, name=name, encoding=enc, auto_save=False)

        self.engine.save_encodings()

        # Update database
        self.db.add_student(
            student_id=student_id,
            name=name,
            roll_no=roll_no,
            branch=branch,
            semester=semester,
        )

        print(f"[Enrollment Complete] {name} successfully enrolled with {num_samples} samples.")
        return True

    def scan_and_rebuild_from_dataset(self) -> int:
        """
        Scans `data/students/` directory, extracts encodings for all images,
        and rebuilds `encodings.pkl` and SQLite database.
        """
        count = 0
        if not config.STUDENTS_DIR.exists():
            return 0

        self.engine.known_encodings.clear()
        self.engine.known_ids.clear()
        self.engine.known_names.clear()

        for folder in sorted(config.STUDENTS_DIR.iterdir()):
            if not folder.is_dir():
                continue

            folder_name = folder.name
            # Parse folder name format: "STU001_Aarav_Sharma"
            if "_" in folder_name:
                parts = folder_name.split("_", 1)
                student_id = parts[0]
                name = parts[1].replace("_", " ")
            else:
                student_id = folder_name
                name = folder_name

            roll_no = f"21BCSE{student_id.replace('STU', '')}"

            image_files = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            if not image_files:
                continue

            success = self.enroll_from_images(
                student_id=student_id,
                name=name,
                roll_no=roll_no,
                image_paths=image_files,
            )
            if success:
                count += 1

        print(f"[Rebuild Complete] Indexed {count} students into database.")
        return count
