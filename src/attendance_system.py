"""
Core Real-Time Attendance System.
Processes live camera streams, classroom panorama photos, and video files.
Draws visual bounding boxes, HUD telemetry, and automatically logs attendance
tagged subject-wise according to the academic timetable (Monday-Saturday, 10 AM - 4 PM).
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import cv2
import numpy as np

import config
from src.database import DatabaseManager
from src.face_engine import FaceEngine, DetectedFace
from src.liveness import LivenessDetector
from src.schedule_manager import ScheduleManager


class AttendanceSystem:
    """Classroom Attendance Controller orchestrating detection, verification, and subject logging."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        engine: Optional[FaceEngine] = None,
        course_code: str = "AUTO",
        session_name: str = config.DEFAULT_SESSION_NAME,
        enable_liveness: bool = False,
    ):
        self.db = db or DatabaseManager()
        self.engine = engine or FaceEngine()
        self.schedule_mgr = ScheduleManager()
        self.course_code = course_code
        self.session_name = session_name
        self.enable_liveness = enable_liveness

        # Debounce tracker: {(student_id, course_code): last_logged_timestamp}
        self.last_marked_time: Dict[tuple, float] = {}
        # Notification banner state
        self.notification_msg: str = "System Ready. Scanning for student attendance..."
        self.notification_expiry: float = time.time() + 4.0
        self.notification_color = (0, 220, 0)

    def resolve_current_course(self, dt: Optional[datetime] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Determines the current course code, course name, and session type based strictly
        on the active lecture in the academic timetable.
        Returns (course_code, course_name, session_type) if a lecture is active, else (None, None, None).
        """
        now = dt or datetime.now()
        active = self.schedule_mgr.get_active_slot(now)
        if active:
            return active["course_code"], active["course_name"], active["session_type"]
        return None, None, None

    def trigger_attendance(self, face: DetectedFace, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Evaluates and records attendance for a recognized student.
        Strictly enforces that a lecture must be currently active.
        """
        if not face.is_known or not face.student_id:
            return False, "Unknown face"

        now_dt = dt or datetime.now()
        active_slot = self.schedule_mgr.get_active_slot(now_dt)

        # 1. Reject immediately if no lecture is scheduled
        if not active_slot:
            self.notification_msg = "Attendance is currently unavailable. No lecture is scheduled at this time."
            self.notification_expiry = time.time() + 3.5
            self.notification_color = (0, 70, 255)  # Red warning
            return False, self.notification_msg

        c_code = active_slot["course_code"]
        s_name = active_slot["session_type"]
        now = time.time()
        debounce_key = (face.student_id, c_code)
        last_time = self.last_marked_time.get(debounce_key, 0.0)

        # Debounce check to avoid spamming within debounce interval
        if (now - last_time) < config.DEBOUNCE_SECONDS:
            return False, f"Debounced for {face.student_id} in {c_code}"

        success, message = self.db.mark_attendance(
            student_id=face.student_id,
            name=face.name,
            confidence=face.confidence,
            course_code=c_code,
            session_name=s_name,
            dt=now_dt,
        )

        self.last_marked_time[debounce_key] = now
        self.notification_msg = message
        self.notification_expiry = now + 4.5
        self.notification_color = (0, 220, 0) if success else (0, 165, 255)
        print(f"[Attendance] {message}")
        return success, message

    def draw_hud(
        self,
        frame: np.ndarray,
        faces: List[DetectedFace],
        fps: float,
        dt: Optional[datetime] = None,
    ) -> np.ndarray:
        """Renders bounding boxes, subject badges, and header/footer telemetry."""
        display = frame.copy()
        h, w = display.shape[:2]
        now_dt = dt or datetime.now()
        active_slot = self.schedule_mgr.get_active_slot(now_dt)
        sched_status = self.schedule_mgr.get_schedule_status(now_dt)

        # 1. Draw Bounding Boxes and Name Badges for each face
        for face in faces:
            left, top, right, bottom = face.left, face.top, face.right, face.bottom

            if face.is_known:
                if active_slot:
                    color = config.COLOR_KNOWN
                    label = f"{face.name} ({face.confidence:.1f}%)"
                    sub_label = f"ID: {face.student_id} | {active_slot['course_code']}"
                else:
                    color = (0, 165, 255)  # Orange: attendance disabled
                    label = f"{face.name} ({face.confidence:.1f}%)"
                    sub_label = "Attendance Disabled (No Lecture)"
            else:
                color = config.COLOR_UNKNOWN
                label = "Unknown Person"
                sub_label = "Not Registered"

            # Corner brackets around face for modern tech aesthetic
            cv2.rectangle(display, (left, top), (right, bottom), color, 2)
            corner_len = int(min(right - left, bottom - top) * 0.25)
            # Top-left corner
            cv2.line(display, (left, top), (left + corner_len, top), color, 4)
            cv2.line(display, (left, top), (left, top + corner_len), color, 4)
            # Top-right corner
            cv2.line(display, (right, top), (right - corner_len, top), color, 4)
            cv2.line(display, (right, top), (right, top + corner_len), color, 4)
            # Bottom-left corner
            cv2.line(display, (left, bottom), (left + corner_len, bottom), color, 4)
            cv2.line(display, (left, bottom), (left, bottom - corner_len), color, 4)
            # Bottom-right corner
            cv2.line(display, (right, bottom), (right - corner_len, bottom), color, 4)
            cv2.line(display, (right, bottom), (right, bottom - corner_len), color, 4)

            # Draw name tag badge above the face
            tag_height = 42
            tag_top = max(0, top - tag_height)
            tag_bottom = top
            tag_width = max(220, right - left)

            cv2.rectangle(
                display,
                (left, tag_top),
                (left + tag_width, tag_bottom),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                display,
                label,
                (left + 6, tag_top + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                display,
                sub_label,
                (left + 6, tag_top + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (20, 20, 20),
                1,
            )

        # 2. Top Header Telemetry Bar
        cv2.rectangle(display, (0, 0), (w, 54), config.COLOR_BANNER, cv2.FILLED)
        day_str = now_dt.strftime("%A, %b %d")

        if active_slot:
            title_text = f"ACTIVE LECTURE: {active_slot['course_code']} \u2014 {active_slot['course_name'].upper()}"
            sub_text = f"{day_str} | Slot: {active_slot['start']} - {active_slot['end']} | Room: {active_slot['room']} | Faculty: {active_slot['faculty']}"
            title_color = (0, 255, 200)
            sub_color = (0, 220, 0)
        else:
            title_text = "NO ACTIVE LECTURE \u2014 ATTENDANCE DISABLED"
            sub_text = f"{day_str} | {sched_status['message']}"
            title_color = (0, 70, 255)  # Warning Red
            sub_color = (0, 165, 255)  # Orange

        cv2.putText(display, title_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, title_color, 2)
        cv2.putText(display, sub_text, (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sub_color, 1)

        # Clock & FPS
        time_str = now_dt.strftime("%H:%M:%S")
        fps_text = f"FPS: {fps:.1f} | {time_str}"
        cv2.putText(display, fps_text, (w - 240, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        # 3. Bottom Notification Bar
        now = time.time()
        if now < self.notification_expiry:
            banner_bg = (30, 30, 30)
            cv2.rectangle(display, (0, h - 45), (w, h), banner_bg, cv2.FILLED)
            cv2.putText(
                display,
                f">> {self.notification_msg}",
                (20, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.notification_color,
                2,
            )
        else:
            # Persistent status
            cv2.rectangle(display, (0, h - 35), (w, h), (20, 20, 20), cv2.FILLED)
            if active_slot:
                status_text = (
                    f"Active: {active_slot['course_code']} ({active_slot['course_name']}) | "
                    f"One attendance per student per lecture | [Q] Quit  [S] Snapshot  [E] Export"
                )
            else:
                status_text = (
                    f"NO ACTIVE LECTURE \u2014 ATTENDANCE DISABLED | Class Hours: Mon-Sat 10:00 AM - 04:00 PM | "
                    f"[Q] Quit  [S] Snapshot  [E] Export"
                )
            cv2.putText(display, status_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        return display

    def run_live(self, camera_index: int = config.CAMERA_INDEX) -> None:
        """
        Runs the real-time camera loop for classroom attendance.
        Press 'q' to exit, 's' to capture a snapshot, 'e' to export CSV logs immediately.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[Error] Failed to open camera device #{camera_index}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        active = self.schedule_mgr.get_active_slot()
        status_info = self.schedule_mgr.get_schedule_status()

        print("\n=======================================================")
        print(" Live Attendance Scanner Started")
        if active:
            print(f" Status: ACTIVE LECTURE")
            print(f" Subject: {active['course_code']} - {active['course_name']} ({active['session_type']})")
            print(f" Slot Time: {active['start']} - {active['end']} | Room: {active['room']}")
        else:
            print(f" Status: NO ACTIVE LECTURE — ATTENDANCE DISABLED")
            print(f" Message: {status_info['message']}")
        print(f" Timetable: Monday to Saturday | 10:00 AM - 04:00 PM")
        print(f" Loaded {self.engine.total_registered} registered student profiles")
        print(" Controls: [Q] Quit | [S] Save Snapshot | [E] Export CSV")
        print("=======================================================\n")

        prev_time = time.time()
        frame_count = 0
        current_faces: List[DetectedFace] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Blank frame received or camera disconnected.")
                break

            frame_count += 1
            now = time.time()
            fps = 1.0 / max(1e-4, now - prev_time)
            prev_time = now

            # Process face detection every N frames for smooth video
            if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                current_faces = self.engine.process_frame(frame, scale=0.5)

                for face in current_faces:
                    # Optional Liveness Verification
                    if self.enable_liveness and face.is_known:
                        crop = frame[face.top:face.bottom, face.left:face.right]
                        is_live, score = LivenessDetector.check_texture_liveness(crop)
                        if not is_live:
                            face.name = "Spoof Detected"
                            face.student_id = None
                            face.confidence = 0.0
                            continue

                    self.trigger_attendance(face)

            # Render display HUD
            annotated_frame = self.draw_hud(frame, current_faces, fps)
            cv2.imshow("Classroom AI Attendance System", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snap_name = config.DATA_DIR / f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_name), annotated_frame)
                self.notification_msg = f"Snapshot saved to: {snap_name.name}"
                self.notification_expiry = time.time() + 3.0
                self.notification_color = (255, 255, 0)
            elif key == ord('e'):
                csv_path = self.db.export_to_csv()
                self.notification_msg = f"Exported attendance to {csv_path.name}"
                self.notification_expiry = time.time() + 4.0
                self.notification_color = (0, 255, 255)

        cap.release()
        cv2.destroyAllWindows()
        csv_file = self.db.export_to_csv()
        print(f"\n[Session Finished] Attendance records successfully exported to {csv_file}")

    def scan_image(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        dt: Optional[datetime] = None,
    ) -> Dict:
        """
        Scans a classroom photo containing multiple students and logs subject-wise attendance.
        Strictly validates active lecture slot at the time of scan (or injected dt).
        """
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not open image file: {image_path}")

        faces = self.engine.process_frame(bgr, scale=1.0)
        marked_students = []

        for face in faces:
            if face.is_known:
                success, _ = self.trigger_attendance(face, dt=dt)
                if success:
                    marked_students.append({
                        "student_id": face.student_id,
                        "name": face.name,
                        "confidence": face.confidence,
                    })

        annotated = self.draw_hud(bgr, faces, fps=0.0, dt=dt)

        if output_path is None:
            output_path = image_path.parent / f"annotated_{image_path.name}"

        cv2.imwrite(str(output_path), annotated)

        return {
            "total_faces_detected": len(faces),
            "marked_students": marked_students,
            "annotated_image": output_path,
        }
