"""
Automated Test Suite for Timetable-Based Attendance Validation.
Tests active slot detection, interval boundaries (start-inclusive, end-exclusive),
transit gaps, lunch breaks, Sundays, database insertion, and duplicate prevention.
"""
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.schedule_manager import ScheduleManager, WEEKLY_SCHEDULE, CURRICULUM_COURSES
from src.database import DatabaseManager


class TestTimetableActiveSlotResolution(unittest.TestCase):
    """Tests schedule resolution and interval boundaries against WEEKLY_SCHEDULE."""

    def setUp(self):
        self.mgr = ScheduleManager()

    def test_interval_boundaries_monday(self):
        # 1. Before 10:00 AM -> None
        t_before = datetime(2026, 11, 23, 9, 59, 59)  # Monday
        self.assertIsNone(self.mgr.get_active_slot(t_before))

        # 2. Exact Slot 1 Start: 10:00:00 -> CS301 (Start-inclusive)
        t_slot1_start = datetime(2026, 11, 23, 10, 0, 0)
        s1 = self.mgr.get_active_slot(t_slot1_start)
        self.assertIsNotNone(s1)
        self.assertEqual(s1["course_code"], "CS301")
        self.assertEqual(s1["start"], "10:00")
        self.assertEqual(s1["end"], "11:00")

        # 3. Middle of Slot 1: 10:30:00 -> CS301
        t_slot1_mid = datetime(2026, 11, 23, 10, 30, 0)
        s1_mid = self.mgr.get_active_slot(t_slot1_mid)
        self.assertIsNotNone(s1_mid)
        self.assertEqual(s1_mid["course_code"], "CS301")

        # 4. 1 second before Slot 1 End: 10:59:59 -> CS301
        t_slot1_end_minus1 = datetime(2026, 11, 23, 10, 59, 59)
        s1_end = self.mgr.get_active_slot(t_slot1_end_minus1)
        self.assertIsNotNone(s1_end)
        self.assertEqual(s1_end["course_code"], "CS301")

        # 5. Exact Slot 1 End: 11:00:00 -> None (End-exclusive transit gap)
        t_slot1_exact_end = datetime(2026, 11, 23, 11, 0, 0)
        self.assertIsNone(self.mgr.get_active_slot(t_slot1_exact_end))

        # 6. Transit Gap between Slot 1 and Slot 2: 11:10:00 -> None
        t_gap1 = datetime(2026, 11, 23, 11, 10, 0)
        self.assertIsNone(self.mgr.get_active_slot(t_gap1))

        # 7. Exact Slot 2 Start: 11:15:00 -> CS302
        t_slot2_start = datetime(2026, 11, 23, 11, 15, 0)
        s2 = self.mgr.get_active_slot(t_slot2_start)
        self.assertIsNotNone(s2)
        self.assertEqual(s2["course_code"], "CS302")

        # 8. Lunch Break: 12:15:00 to 13:00:00 -> None
        t_lunch_start = datetime(2026, 11, 23, 12, 15, 0)
        self.assertIsNone(self.mgr.get_active_slot(t_lunch_start))
        t_lunch_mid = datetime(2026, 11, 23, 12, 45, 0)
        self.assertIsNone(self.mgr.get_active_slot(t_lunch_mid))
        t_lunch_end_minus1 = datetime(2026, 11, 23, 12, 59, 59)
        self.assertIsNone(self.mgr.get_active_slot(t_lunch_end_minus1))

        # 9. Slot 3 Start: 13:00:00 -> CS303
        t_slot3_start = datetime(2026, 11, 23, 13, 0, 0)
        s3 = self.mgr.get_active_slot(t_slot3_start)
        self.assertIsNotNone(s3)
        self.assertEqual(s3["course_code"], "CS303")

        # 10. Transit Gap: 14:00:00 to 14:15:00 -> None
        t_gap2 = datetime(2026, 11, 23, 14, 5, 0)
        self.assertIsNone(self.mgr.get_active_slot(t_gap2))

        # 11. Slot 4 (Lab): 14:15:00 to 16:00:00 -> CS305
        t_slot4_start = datetime(2026, 11, 23, 14, 15, 0)
        s4 = self.mgr.get_active_slot(t_slot4_start)
        self.assertIsNotNone(s4)
        self.assertEqual(s4["course_code"], "CS305")

        t_slot4_end_minus1 = datetime(2026, 11, 23, 15, 59, 59)
        s4_end = self.mgr.get_active_slot(t_slot4_end_minus1)
        self.assertIsNotNone(s4_end)
        self.assertEqual(s4_end["course_code"], "CS305")

        # 12. At and after 16:00:00 -> None
        t_close = datetime(2026, 11, 23, 16, 0, 0)
        self.assertIsNone(self.mgr.get_active_slot(t_close))
        t_after = datetime(2026, 11, 23, 17, 30, 0)
        self.assertIsNone(self.mgr.get_active_slot(t_after))

    def test_saturday_schedule(self):
        # Saturday: 2026-11-28
        # Slot 1: 10:00 - 11:30 -> CS305
        s1 = self.mgr.get_active_slot(datetime(2026, 11, 28, 10, 15, 0))
        self.assertIsNotNone(s1)
        self.assertEqual(s1["course_code"], "CS305")

        # Transit Gap: 11:30:00 -> None
        self.assertIsNone(self.mgr.get_active_slot(datetime(2026, 11, 28, 11, 30, 0)))

        # Slot 2: 11:45 - 13:15 -> CS301
        s2 = self.mgr.get_active_slot(datetime(2026, 11, 28, 12, 0, 0))
        self.assertIsNotNone(s2)
        self.assertEqual(s2["course_code"], "CS301")

        # Lunch Break: 13:15 - 14:00 -> None
        self.assertIsNone(self.mgr.get_active_slot(datetime(2026, 11, 28, 13, 30, 0)))

        # Slot 3: 14:00 - 16:00 -> CS304
        s3 = self.mgr.get_active_slot(datetime(2026, 11, 28, 14, 30, 0))
        self.assertIsNotNone(s3)
        self.assertEqual(s3["course_code"], "CS304")

        # After 16:00 -> None
        self.assertIsNone(self.mgr.get_active_slot(datetime(2026, 11, 28, 16, 0, 0)))

    def test_sunday_non_working_day(self):
        # Sunday: 2026-11-29
        t_sunday = datetime(2026, 11, 29, 10, 30, 0)
        self.assertIsNone(self.mgr.get_active_slot(t_sunday))
        status = self.mgr.get_schedule_status(t_sunday)
        self.assertFalse(status["is_active"])
        self.assertIn("Sunday", status["message"])


class TestDatabaseTimetableAttendanceValidation(unittest.TestCase):
    """Tests SQLite attendance insertion, rejection outside slots, and duplicate handling."""

    def setUp(self):
        self.test_db_path = ROOT_DIR / "data" / "test_attendance_validation.db"
        self.db = DatabaseManager(db_path=self.test_db_path)
        with self.db._get_connection() as conn:
            conn.execute("DELETE FROM attendance")
            conn.execute("DELETE FROM students")
            conn.commit()
        # Register two test students
        self.db.add_student("STU_A", "Alice Sharma", "21CS001", "CSE", 5)
        self.db.add_student("STU_B", "Bob Patel", "21CS002", "CSE", 5)

    def tearDown(self):
        try:
            with self.db._get_connection() as conn:
                conn.execute("DELETE FROM attendance")
                conn.execute("DELETE FROM students")
                conn.commit()
        except Exception:
            pass

    def test_attendance_allowed_during_active_lecture(self):
        # Monday 10:15 AM -> Active CS301 (Computer Vision)
        dt_active = datetime(2026, 11, 23, 10, 15, 0)
        success, msg = self.db.mark_attendance(
            student_id="STU_A",
            name="Alice Sharma",
            confidence=95.5,
            dt=dt_active,
        )
        self.assertTrue(success)
        self.assertIn("CS301", msg)

        # Verify DB record
        records = self.db.get_attendance_records(date_str="2026-11-23", course_code="CS301")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["student_id"], "STU_A")
        self.assertEqual(records[0]["course_code"], "CS301")

    def test_attendance_rejected_when_no_active_lecture(self):
        # Monday 11:05 AM -> Transit gap (No lecture)
        dt_gap = datetime(2026, 11, 23, 11, 5, 0)
        success, msg = self.db.mark_attendance(
            student_id="STU_A",
            name="Alice Sharma",
            confidence=94.0,
            dt=dt_gap,
        )
        self.assertFalse(success)
        self.assertIn("No lecture is scheduled at this time", msg)

        # Verify no record was inserted
        records = self.db.get_attendance_records(date_str="2026-11-23")
        self.assertEqual(len(records), 0)

    def test_duplicate_attendance_rejected_for_same_lecture(self):
        # Monday 10:10 AM -> Alice marks CS301
        dt1 = datetime(2026, 11, 23, 10, 10, 0)
        success1, _ = self.db.mark_attendance(
            student_id="STU_A",
            name="Alice Sharma",
            confidence=96.0,
            dt=dt1,
        )
        self.assertTrue(success1)

        # Monday 10:45 AM -> Alice tries to mark CS301 again (same lecture slot)
        dt2 = datetime(2026, 11, 23, 10, 45, 0)
        success2, msg2 = self.db.mark_attendance(
            student_id="STU_A",
            name="Alice Sharma",
            confidence=95.0,
            dt=dt2,
        )
        self.assertFalse(success2)
        self.assertIn("Already marked for CS301", msg2)

        # Verify only 1 record exists in DB
        records = self.db.get_attendance_records(date_str="2026-11-23", course_code="CS301")
        self.assertEqual(len(records), 1)

    def test_multiple_distinct_lectures_on_same_day(self):
        # Same student, same date, but different active subjects:
        # 1. Monday 10:15 AM -> Slot 1 is CS301
        dt_slot1 = datetime(2026, 11, 23, 10, 15, 0)
        s1, _ = self.db.mark_attendance("STU_A", "Alice Sharma", 95.0, dt=dt_slot1)
        self.assertTrue(s1)

        # 2. Monday 11:30 AM -> Slot 2 is CS302 (Operating Systems)
        dt_slot2 = datetime(2026, 11, 23, 11, 30, 0)
        s2, _ = self.db.mark_attendance("STU_A", "Alice Sharma", 93.0, dt=dt_slot2)
        self.assertTrue(s2)

        # 3. Monday 13:20 PM -> Slot 3 is CS303 (Database Management Systems)
        dt_slot3 = datetime(2026, 11, 23, 13, 20, 0)
        s3, _ = self.db.mark_attendance("STU_A", "Alice Sharma", 94.5, dt=dt_slot3)
        self.assertTrue(s3)

        # Verify DB contains all 3 distinct records for Alice on 2026-11-23
        records = self.db.get_attendance_records(date_str="2026-11-23", student_id="STU_A")
        self.assertEqual(len(records), 3)
        codes = [r["course_code"] for r in records]
        self.assertIn("CS301", codes)
        self.assertIn("CS302", codes)
        self.assertIn("CS303", codes)

    def test_multiple_students_in_same_lecture(self):
        # Monday 10:15 AM -> Alice marks CS301
        dt1 = datetime(2026, 11, 23, 10, 15, 0)
        s1, _ = self.db.mark_attendance("STU_A", "Alice Sharma", 95.0, dt=dt1)
        self.assertTrue(s1)

        # Monday 10:20 AM -> Bob marks CS301 in the same lecture
        dt2 = datetime(2026, 11, 23, 10, 20, 0)
        s2, _ = self.db.mark_attendance("STU_B", "Bob Patel", 92.0, dt=dt2)
        self.assertTrue(s2)

        records = self.db.get_attendance_records(date_str="2026-11-23", course_code="CS301")
        self.assertEqual(len(records), 2)

    def test_mismatched_course_request_rejected(self):
        # Monday 10:15 AM active lecture is CS301.
        # If an external request attempts to force CS304:
        dt = datetime(2026, 11, 23, 10, 15, 0)
        success, msg = self.db.mark_attendance(
            student_id="STU_A",
            name="Alice Sharma",
            confidence=95.0,
            course_code="CS304",
            dt=dt,
        )
        self.assertFalse(success)
        self.assertIn("does not match active scheduled lecture", msg)

    def test_class_headcount_and_who_was_present(self):
        # Register a 3rd enrolled student: Charlie
        self.db.add_student("STU_C", "Charlie Singh", "21CS003", "CSE", 5)

        # In Monday 10:15 AM lecture for CS301:
        # Alice and Bob attend; Charlie is absent.
        dt = datetime(2026, 11, 23, 10, 15, 0)
        self.db.mark_attendance("STU_A", "Alice Sharma", 96.0, dt=dt)
        self.db.mark_attendance("STU_B", "Bob Patel", 94.0, dt=dt)

        # 1. Query class attendance summary
        summaries = self.db.get_class_attendance_summary(date_str="2026-11-23", course_code="CS301")
        self.assertEqual(len(summaries), 1)
        cs = summaries[0]

        # 2. Check "how many" were present
        self.assertEqual(cs["total_present"], 2)
        self.assertEqual(cs["total_enrolled"], 3)
        self.assertEqual(cs["total_absent"], 1)
        self.assertEqual(cs["attendance_percentage"], 66.7)

        # 3. Check "who" was present
        present_names = [p["name"] for p in cs["present_students"]]
        present_rolls = [p["roll_no"] for p in cs["present_students"]]
        self.assertIn("Alice Sharma", present_names)
        self.assertIn("Bob Patel", present_names)
        self.assertIn("21CS001", present_rolls)
        self.assertIn("21CS002", present_rolls)

        # 4. Check "who" was absent
        absent_names = [a["name"] for a in cs["absent_students"]]
        self.assertEqual(absent_names, ["Charlie Singh"])

        # 5. Verify CSV export of class roster
        csv_file = self.db.export_class_roster_csv(date_str="2026-11-23", course_code="CS301")
        self.assertTrue(csv_file.exists())
        content = csv_file.read_text(encoding="utf-8")
        self.assertIn("Alice Sharma", content)
        self.assertIn("Bob Patel", content)
        self.assertIn("Charlie Singh", content)
        self.assertIn("Present", content)
        self.assertIn("Absent", content)

        # Clean up test export
        if csv_file.exists():
            csv_file.unlink()

    def test_demo_mode_bypasses_schedule_enforcement(self):
        # Sunday 10:30 AM (Normally non-working day -> Rejected in strict mode)
        t_sunday = datetime(2026, 11, 29, 10, 30, 0)

        # 1. Strict mode (enforce_window=True) -> Must be rejected
        strict_success, strict_msg = self.db.mark_attendance(
            student_id="STU_A",
            name="Alice Sharma",
            confidence=95.0,
            dt=t_sunday,
            enforce_window=True,
        )
        self.assertFalse(strict_success)
        self.assertIn("No lecture is scheduled", strict_msg)

        # 2. Demo mode (enforce_window=False) -> Must succeed and mark attendance
        demo_success, demo_msg = self.db.mark_attendance(
            student_id="STU_A",
            name="Alice Sharma",
            confidence=95.0,
            course_code="CS301",
            dt=t_sunday,
            enforce_window=False,
        )
        self.assertTrue(demo_success)
        self.assertIn("Marked", demo_msg)

        # Verify record in DB
        records = self.db.get_attendance_records(date_str="2026-11-29", course_code="CS301")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["student_id"], "STU_A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
