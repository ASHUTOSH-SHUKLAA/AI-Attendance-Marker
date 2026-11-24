"""
Database Manager for AI-Powered Attendance System.
Handles SQLite storage, curriculum courses, timetable schedules,
subject-wise attendance tracking, and comprehensive review reports.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

import config
from src.schedule_manager import ScheduleManager, CURRICULUM_COURSES


class DatabaseManager:
    """Manages SQLite database operations, subject-wise attendance logs, and reports."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DATABASE_FILE
        self.schedule_mgr = ScheduleManager()
        self.init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self) -> None:
        """Initializes tables for students, courses, and subject-wise attendance."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Students Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    roll_no TEXT UNIQUE NOT NULL,
                    branch TEXT DEFAULT 'CSE',
                    semester INTEGER DEFAULT 5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # 2. Courses / Curriculum Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS courses (
                    course_code TEXT PRIMARY KEY,
                    course_name TEXT NOT NULL,
                    short_name TEXT,
                    faculty TEXT,
                    credits INTEGER DEFAULT 3,
                    room TEXT DEFAULT 'Classroom'
                )
                """
            )

            # 3. Attendance Table (Subject-wise & Date-wise)
            # Enforces UNIQUE(student_id, date, course_code) so attendance is marked once per subject per day
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    course_code TEXT NOT NULL,
                    course_name TEXT NOT NULL,
                    session_name TEXT NOT NULL,
                    day_of_week TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT DEFAULT 'Present',
                    FOREIGN KEY (student_id) REFERENCES students(student_id),
                    FOREIGN KEY (course_code) REFERENCES courses(course_code),
                    UNIQUE(student_id, date, course_code)
                )
                """
            )

            # Schema Migration: ensure course_name and day_of_week exist if table was created in an older version
            cursor.execute("PRAGMA table_info(attendance)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            if "course_name" not in existing_cols and len(existing_cols) > 0:
                cursor.execute("ALTER TABLE attendance ADD COLUMN course_name TEXT DEFAULT ''")
            if "day_of_week" not in existing_cols and len(existing_cols) > 0:
                cursor.execute("ALTER TABLE attendance ADD COLUMN day_of_week TEXT DEFAULT ''")

            # Purge any mock/synthetic student profiles from database
            mock_names = (
                'Aarav Sharma', 'Ananya Verma', 'Rohan Patel', 'Priya Singh',
                'Ishaan Gupta', 'Sneha Reddy', 'Vikram Malhotra', 'Neha Joshi',
                'Arjun Nair', 'Tanvi Kapoor'
            )
            placeholders = ",".join("?" * len(mock_names))
            cursor.execute(f"DELETE FROM students WHERE name IN ({placeholders})", mock_names)
            cursor.execute(f"DELETE FROM attendance WHERE name IN ({placeholders})", mock_names)

            # Populate courses table with standard 5th semester curriculum
            for c in CURRICULUM_COURSES:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO courses (course_code, course_name, short_name, faculty, credits, room)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (c["course_code"], c["course_name"], c["short_name"], c["faculty"], c["credits"], c["room"]),
                )

            conn.commit()

    def add_student(
        self,
        student_id: str,
        name: str,
        roll_no: str,
        branch: str = "CSE",
        semester: int = 5,
    ) -> bool:
        """Adds or updates a student profile in the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO students (student_id, name, roll_no, branch, semester)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(student_id) DO UPDATE SET
                        name=excluded.name,
                        roll_no=excluded.roll_no,
                        branch=excluded.branch,
                        semester=excluded.semester
                    """,
                    (student_id, name, roll_no, branch, semester),
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"[Database Error] Failed to add student {student_id}: {e}")
            return False

    def get_all_students(self) -> List[Dict]:
        """Returns all registered student records ordered by roll number."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students ORDER BY roll_no ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_student(self, student_id: str) -> Optional[Dict]:
        """Retrieves a single student by student_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_courses(self) -> List[Dict]:
        """Returns list of all curriculum courses."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM courses ORDER BY course_code ASC")
            return [dict(row) for row in cursor.fetchall()]

    def is_already_marked(
        self,
        student_id: str,
        date_str: str,
        course_code: str,
    ) -> bool:
        """
        Checks if attendance has already been recorded for this student, date, and course.
        Enforces one attendance per student per subject per day.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 1 FROM attendance
                WHERE student_id = ? AND date = ? AND course_code = ?
                """,
                (student_id, date_str, course_code),
            )
            return cursor.fetchone() is not None

    def mark_attendance(
        self,
        student_id: str,
        name: str,
        confidence: float,
        course_code: Optional[str] = None,
        session_name: Optional[str] = None,
        dt: Optional[datetime] = None,
        enforce_window: bool = config.ENFORCE_TIME_WINDOW,
    ) -> Tuple[bool, str]:
        """
        Records subject-wise attendance for a student on the current date.
        Strictly validates that a lecture is currently active according to the timetable.
        - Resolves active curriculum course from schedule manager using current time (or injected dt).
        - Rejects attendance if no lecture is currently scheduled.
        - Prevents duplicate attendance for the same subject on the same day.
        """
        now = dt or datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        day_of_week = now.strftime("%A")

        # 1. Resolve Active Scheduled Lecture
        active_slot = self.schedule_mgr.get_active_slot(now)
        if not active_slot:
            return False, "Attendance is currently unavailable. No lecture is scheduled at this time."

        # 2. Validate course code if explicitly requested
        if course_code and course_code != "AUTO" and course_code != active_slot["course_code"]:
            return False, (
                f"Attendance Rejected: Requested course '{course_code}' does not match active scheduled lecture "
                f"({active_slot['course_code']} - {active_slot['course_name']})."
            )

        resolved_course_code = active_slot["course_code"]
        resolved_course_name = active_slot["course_name"]
        resolved_session_name = session_name or active_slot["session_type"]

        # 3. Check duplicate for this day and subject
        if self.is_already_marked(student_id, date_str, resolved_course_code):
            return False, f"Already marked for {resolved_course_code} ({resolved_course_name}) today ({date_str})."

        # 4. Insert Attendance Record
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO attendance
                    (student_id, name, date, time, course_code, course_name, session_name, day_of_week, confidence, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Present')
                    """,
                    (
                        student_id,
                        name,
                        date_str,
                        time_str,
                        resolved_course_code,
                        resolved_course_name,
                        resolved_session_name,
                        day_of_week,
                        round(confidence, 2),
                    ),
                )
                conn.commit()

            # Auto-sync to daily CSV log and class roster (Present & Absent records)
            self.export_to_csv(date_str=date_str, course_code=resolved_course_code)
            try:
                self.export_class_roster_csv(date_str=date_str, course_code=resolved_course_code)
            except Exception:
                pass
            return True, f"Marked: {name} | {resolved_course_code} ({resolved_course_name}) | {time_str}"
        except sqlite3.IntegrityError:
            return False, f"Duplicate prevented: {name} already logged for {resolved_course_code} today."
        except Exception as e:
            return False, f"Database error: {e}"

    def get_attendance_records(
        self,
        date_str: Optional[str] = None,
        course_code: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieves attendance rows with flexible filtering by date, subject, or student."""
        query = "SELECT * FROM attendance WHERE 1=1"
        params = []

        if date_str:
            query += " AND date = ?"
            params.append(date_str)
        if course_code and course_code != "ALL":
            query += " AND course_code = ?"
            params.append(course_code)
        if student_id and student_id != "ALL":
            query += " AND student_id = ?"
            params.append(student_id)

        query += " ORDER BY date DESC, time ASC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_attendance_for_date(
        self,
        date_str: Optional[str] = None,
        course_code: Optional[str] = None,
    ) -> List[Dict]:
        """Convenience method for current date."""
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        return self.get_attendance_records(date_str=date_str, course_code=course_code)

    def get_subject_wise_summary(self) -> List[Dict]:
        """
        Generates comprehensive subject-wise and curriculum-wide review metrics.
        Calculates:
        - Attendance count per subject (CS301, CS302, CS303, CS304, CS305)
        - Total classes attended & total sessions held
        - Overall attendance %
        - Shortage / Defaulter warning if below 75%
        """
        students = self.get_all_students()
        courses = self.get_all_courses()
        course_codes = [c["course_code"] for c in courses]

        # Count total distinct sessions held per course
        classes_held_per_course = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for code in course_codes:
                cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance WHERE course_code = ?", (code,))
                count = cursor.fetchone()[0]
                classes_held_per_course[code] = max(1, count)  # minimum baseline of 1 to prevent div by zero

        total_curriculum_sessions = sum(classes_held_per_course.values())

        summary = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for s in students:
                s_id = s["student_id"]
                row_data = {
                    "student_id": s_id,
                    "name": s["name"],
                    "roll_no": s["roll_no"],
                    "branch": s["branch"],
                    "subjects": {},
                    "total_attended": 0,
                    "total_conducted": total_curriculum_sessions,
                    "overall_pct": 0.0,
                    "is_defaulter": False,
                }

                total_attended = 0
                for code in course_codes:
                    cursor.execute(
                        "SELECT COUNT(*) FROM attendance WHERE student_id = ? AND course_code = ?",
                        (s_id, code),
                    )
                    attended = cursor.fetchone()[0]
                    conducted = classes_held_per_course[code]
                    pct = round((attended / conducted * 100), 1) if conducted > 0 else 0.0
                    row_data["subjects"][code] = {
                        "attended": attended,
                        "conducted": conducted,
                        "pct": pct,
                    }
                    total_attended += attended

                row_data["total_attended"] = total_attended
                overall_pct = (
                    round((total_attended / total_curriculum_sessions * 100), 1)
                    if total_curriculum_sessions > 0
                    else 0.0
                )
                row_data["overall_pct"] = overall_pct
                row_data["is_defaulter"] = overall_pct < config.MIN_ATTENDANCE_PERCENTAGE
                summary.append(row_data)

        return summary

    def export_to_csv(
        self,
        date_str: Optional[str] = None,
        course_code: Optional[str] = None,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Exports attendance logs for a date and/or course to CSV."""
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        rows = self.get_attendance_records(date_str=date_str, course_code=course_code)

        if output_file is None:
            prefix = f"{course_code}_" if course_code and course_code != "ALL" else ""
            filename = f"attendance_{prefix}{date_str}.csv"
            output_file = config.ATTENDANCE_LOGS_DIR / filename

        columns_order = [
            "id",
            "student_id",
            "name",
            "date",
            "time",
            "course_code",
            "course_name",
            "session_name",
            "day_of_week",
            "confidence",
            "status",
        ]

        if rows:
            df = pd.DataFrame(rows)
            available_cols = [c for c in columns_order if c in df.columns]
            df = df[available_cols]
        else:
            df = pd.DataFrame(columns=columns_order)

        df.to_csv(str(output_file), index=False)
        return output_file

    def get_class_attendance_summary(
        self,
        date_str: Optional[str] = None,
        course_code: Optional[str] = None,
    ) -> List[Dict]:
        """
        Maintains and returns class-by-class records of:
        - How many were present (total_present, total_enrolled, total_absent, attendance_percentage)
        - Who were present (detailed list of attendees with roll numbers, marked timestamps, confidence)
        - Who were absent (enrolled students who did not attend)
        """
        all_students = self.get_all_students()
        student_map = {s["student_id"]: s for s in all_students}
        total_enrolled = len(all_students)

        query = """
            SELECT DISTINCT date, course_code, course_name, session_name, day_of_week
            FROM attendance
            WHERE 1=1
        """
        params = []
        if date_str:
            query += " AND date = ?"
            params.append(date_str)
        if course_code and course_code != "ALL":
            query += " AND course_code = ?"
            params.append(course_code)

        query += " ORDER BY date DESC, course_code ASC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            class_sessions = [dict(row) for row in cursor.fetchall()]

            summaries = []
            for session in class_sessions:
                d = session["date"]
                c_code = session["course_code"]

                # Fetch all attendance rows for this specific class
                cursor.execute(
                    """
                    SELECT a.student_id, a.name, a.time, a.confidence, a.status,
                           COALESCE(s.roll_no, 'N/A') as roll_no,
                           COALESCE(s.branch, 'CSE') as branch
                    FROM attendance a
                    LEFT JOIN students s ON a.student_id = s.student_id
                    WHERE a.date = ? AND a.course_code = ?
                    ORDER BY a.time ASC
                    """,
                    (d, c_code),
                )
                present_rows = [dict(r) for r in cursor.fetchall()]
                present_ids = {r["student_id"] for r in present_rows}

                # Construct list of present students (Who was present)
                present_students = []
                for r in present_rows:
                    present_students.append({
                        "student_id": r["student_id"],
                        "name": r["name"],
                        "roll_no": r["roll_no"],
                        "branch": r["branch"],
                        "time": r["time"],
                        "confidence": r["confidence"],
                        "status": r["status"],
                    })

                # Construct list of absent students (Who was absent)
                absent_students = []
                for s in all_students:
                    if s["student_id"] not in present_ids:
                        absent_students.append({
                            "student_id": s["student_id"],
                            "name": s["name"],
                            "roll_no": s["roll_no"],
                            "branch": s["branch"],
                            "status": "Absent",
                        })

                total_present = len(present_students)
                total_absent = len(absent_students)
                pct = round((total_present / total_enrolled * 100), 1) if total_enrolled > 0 else 0.0

                summaries.append({
                    "date": d,
                    "course_code": c_code,
                    "course_name": session["course_name"],
                    "session_name": session["session_name"],
                    "day_of_week": session["day_of_week"],
                    "total_present": total_present,
                    "total_enrolled": total_enrolled,
                    "total_absent": total_absent,
                    "attendance_percentage": pct,
                    "present_students": present_students,
                    "absent_students": absent_students,
                })

            return summaries

    def get_class_roster_for_session(
        self,
        date_str: str,
        course_code: str,
    ) -> Optional[Dict]:
        """Returns detailed roster (headcount, attendees, and absentees) for a specific class session."""
        summaries = self.get_class_attendance_summary(date_str=date_str, course_code=course_code)
        return summaries[0] if summaries else None

    def export_class_roster_csv(
        self,
        date_str: str,
        course_code: str,
        output_file: Optional[Path] = None,
    ) -> Path:
        """
        Exports a complete class attendance roster (both present and absent students) to CSV.
        Maintains who was present, who was absent, timestamps, and confidence scores.
        """
        roster = self.get_class_roster_for_session(date_str, course_code)
        if output_file is None:
            filename = f"class_roster_{course_code}_{date_str}.csv"
            output_file = config.ATTENDANCE_LOGS_DIR / filename

        rows = []
        if roster:
            for p in roster["present_students"]:
                rows.append({
                    "Roll No": p["roll_no"],
                    "Student ID": p["student_id"],
                    "Name": p["name"],
                    "Branch": p["branch"],
                    "Status": "Present",
                    "Time Marked": p["time"],
                    "Confidence": f"{p['confidence']}%",
                    "Course Code": roster["course_code"],
                    "Course Name": roster["course_name"],
                    "Session Type": roster["session_name"],
                    "Date": roster["date"],
                    "Day": roster["day_of_week"],
                })
            for a in roster["absent_students"]:
                rows.append({
                    "Roll No": a["roll_no"],
                    "Student ID": a["student_id"],
                    "Name": a["name"],
                    "Branch": a["branch"],
                    "Status": "Absent",
                    "Time Marked": "-",
                    "Confidence": "-",
                    "Course Code": roster["course_code"],
                    "Course Name": roster["course_name"],
                    "Session Type": roster["session_name"],
                    "Date": roster["date"],
                    "Day": roster["day_of_week"],
                })

        df = pd.DataFrame(rows)
        df.to_csv(str(output_file), index=False)
        return output_file

    def export_all_class_sessions_csv(self, output_file: Optional[Path] = None) -> Path:
        """Exports a consolidated summary of all conducted classes with headcount and present attendees."""
        sessions = self.get_class_attendance_summary()
        output_file = output_file or (config.ATTENDANCE_LOGS_DIR / "all_classes_headcount_summary.csv")

        flat = []
        for s in sessions:
            attendee_names = "; ".join([f"{p['roll_no']}: {p['name']}" for p in s["present_students"]])
            absentee_names = "; ".join([f"{a['roll_no']}: {a['name']}" for a in s["absent_students"]])
            flat.append({
                "Date": s["date"],
                "Day": s["day_of_week"],
                "Course Code": s["course_code"],
                "Course Name": s["course_name"],
                "Session Type": s["session_name"],
                "Total Present (How Many)": s["total_present"],
                "Total Enrolled": s["total_enrolled"],
                "Total Absent": s["total_absent"],
                "Attendance %": f"{s['attendance_percentage']}%",
                "Present Students (Who Was Present)": attendee_names,
                "Absent Students": absentee_names,
            })

        df = pd.DataFrame(flat)
        df.to_csv(str(output_file), index=False)
        return output_file

    def export_subject_wise_summary_csv(self, output_file: Optional[Path] = None) -> Path:
        """Exports the curriculum subject-wise review table to CSV."""
        summary = self.get_subject_wise_summary()
        output_file = output_file or (config.ATTENDANCE_LOGS_DIR / "curriculum_attendance_review.csv")

        flat_rows = []
        for row in summary:
            item = {
                "Student ID": row["student_id"],
                "Name": row["name"],
                "Roll Number": row["roll_no"],
                "Branch": row["branch"],
            }
            for code, data in row["subjects"].items():
                item[f"{code} ({data['attended']}/{data['conducted']})"] = f"{data['pct']}%"

            item["Total Attended"] = row["total_attended"]
            item["Total Conducted"] = row["total_conducted"]
            item["Overall Percentage"] = f"{row['overall_pct']}%"
            item["Status"] = "SHORTAGE (<75%)" if row["is_defaulter"] else "ELIGIBLE"
            flat_rows.append(item)

        df = pd.DataFrame(flat_rows)
        df.to_csv(str(output_file), index=False)
        return output_file


def reset_all_data(db_path: Optional[Path] = None) -> None:
    """
    Clears all stored students, encodings, and attendance logs.
    Leaves a clean slate ready for registering real students.
    """
    import pickle
    import shutil

    db_file = db_path or config.DATABASE_FILE

    # 1. Reset database tables
    if db_file.exists():
        with sqlite3.connect(str(db_file)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance;")
            cursor.execute("DELETE FROM students;")
            conn.commit()

    # 2. Reset encodings.pkl to clean empty state
    config.ENCODINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": [], "student_ids": [], "names": []}, f)

    # 3. Clean student image folders in data/students/
    if config.STUDENTS_DIR.exists():
        for item in config.STUDENTS_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()

    # 4. Clean attendance logs in data/attendance_logs/
    if config.ATTENDANCE_LOGS_DIR.exists():
        for f in config.ATTENDANCE_LOGS_DIR.glob("*.csv"):
            f.unlink()

    print("[System Reset] Successfully cleared all student records, encodings, and attendance logs.")
