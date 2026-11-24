"""
Curriculum and Timetable Schedule Manager.
Enforces subject-wise, course-wise, and time-slot schedules for attendance.
Supports Monday to Saturday operational hours (10:00 AM - 04:00 PM) with 5th semester curriculum.
"""
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple
import config


# 5th Semester Curriculum Courses
CURRICULUM_COURSES: List[Dict] = [
    {
        "course_code": "CS301",
        "course_name": "Computer Vision",
        "short_name": "CV",
        "faculty": "Dr. R. K. Sharma",
        "credits": 4,
        "room": "Lab 302",
    },
    {
        "course_code": "CS302",
        "course_name": "Operating Systems",
        "short_name": "OS",
        "faculty": "Prof. S. Gupta",
        "credits": 4,
        "room": "LH-101",
    },
    {
        "course_code": "CS303",
        "course_name": "Database Management Systems",
        "short_name": "DBMS",
        "faculty": "Dr. A. Verma",
        "credits": 4,
        "room": "LH-102",
    },
    {
        "course_code": "CS304",
        "course_name": "Computer Networks",
        "short_name": "CN",
        "faculty": "Prof. K. Patel",
        "credits": 3,
        "room": "LH-101",
    },
    {
        "course_code": "CS305",
        "course_name": "Software Engineering & CV Project",
        "short_name": "SE-Project",
        "faculty": "Dr. M. Reddy",
        "credits": 3,
        "room": "Lab 305",
    },
]

# Standard Daily Lecture Slots (Between 10:00 AM and 04:00 PM)
# Monday to Saturday Timetable Matrix
WEEKLY_SCHEDULE: Dict[str, List[Dict]] = {
    "Monday": [
        {"slot_id": 1, "start": "10:00", "end": "11:00", "course_code": "CS301", "session_type": "Lecture"},
        {"slot_id": 2, "start": "11:15", "end": "12:15", "course_code": "CS302", "session_type": "Lecture"},
        {"slot_id": 3, "start": "13:00", "end": "14:00", "course_code": "CS303", "session_type": "Lecture"},
        {"slot_id": 4, "start": "14:15", "end": "16:00", "course_code": "CS305", "session_type": "Lab Session"},
    ],
    "Tuesday": [
        {"slot_id": 1, "start": "10:00", "end": "11:00", "course_code": "CS304", "session_type": "Lecture"},
        {"slot_id": 2, "start": "11:15", "end": "12:15", "course_code": "CS301", "session_type": "Lecture"},
        {"slot_id": 3, "start": "13:00", "end": "14:00", "course_code": "CS302", "session_type": "Lecture"},
        {"slot_id": 4, "start": "14:15", "end": "16:00", "course_code": "CS303", "session_type": "Practical Lab"},
    ],
    "Wednesday": [
        {"slot_id": 1, "start": "10:00", "end": "11:00", "course_code": "CS303", "session_type": "Lecture"},
        {"slot_id": 2, "start": "11:15", "end": "12:15", "course_code": "CS304", "session_type": "Lecture"},
        {"slot_id": 3, "start": "13:00", "end": "14:00", "course_code": "CS301", "session_type": "Lecture"},
        {"slot_id": 4, "start": "14:15", "end": "16:00", "course_code": "CS305", "session_type": "Project Work"},
    ],
    "Thursday": [
        {"slot_id": 1, "start": "10:00", "end": "11:00", "course_code": "CS302", "session_type": "Lecture"},
        {"slot_id": 2, "start": "11:15", "end": "12:15", "course_code": "CS304", "session_type": "Lecture"},
        {"slot_id": 3, "start": "13:00", "end": "14:00", "course_code": "CS303", "session_type": "Lecture"},
        {"slot_id": 4, "start": "14:15", "end": "16:00", "course_code": "CS301", "session_type": "Vision Lab"},
    ],
    "Friday": [
        {"slot_id": 1, "start": "10:00", "end": "11:00", "course_code": "CS301", "session_type": "Lecture"},
        {"slot_id": 2, "start": "11:15", "end": "12:15", "course_code": "CS303", "session_type": "Lecture"},
        {"slot_id": 3, "start": "13:00", "end": "14:00", "course_code": "CS304", "session_type": "Lecture"},
        {"slot_id": 4, "start": "14:15", "end": "16:00", "course_code": "CS302", "session_type": "OS Lab"},
    ],
    "Saturday": [
        {"slot_id": 1, "start": "10:00", "end": "11:30", "course_code": "CS305", "session_type": "Seminar / Review"},
        {"slot_id": 2, "start": "11:45", "end": "13:15", "course_code": "CS301", "session_type": "Tutorial & Viva"},
        {"slot_id": 3, "start": "14:00", "end": "16:00", "course_code": "CS304", "session_type": "Network Lab"},
    ],
}


class ScheduleManager:
    """Manages course curriculum timetable, operational windows, and slot resolution."""

    def __init__(self):
        self.courses_by_code = {c["course_code"]: c for c in CURRICULUM_COURSES}
        self.operating_days = getattr(
            config,
            "SCHEDULE_OPERATING_DAYS",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        )
        self.start_hour = 10  # 10:00 AM
        self.end_hour = 16    # 04:00 PM

    def get_course_details(self, course_code: str) -> Dict:
        """Returns metadata for a course code."""
        return self.courses_by_code.get(
            course_code,
            {
                "course_code": course_code,
                "course_name": "General Attendance",
                "short_name": course_code,
                "faculty": "Assigned Faculty",
                "credits": 3,
                "room": "Auditorium",
            },
        )

    def is_within_operating_window(self, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Validates if current time falls within Monday-Saturday 10:00 AM - 04:00 PM.
        Returns (is_open, status_message).
        """
        now = dt or datetime.now()
        day_name = now.strftime("%A")

        if day_name not in self.operating_days:
            return False, f"Attendance window is CLOSED on {day_name}. (Open: Monday to Saturday)"

        current_time = now.time()
        start_time = dtime(self.start_hour, 0)
        end_time = dtime(self.end_hour, 0)

        if current_time < start_time:
            return False, f"Attendance window opens at 10:00 AM. Current time is {now.strftime('%I:%M %p')}."
        elif current_time >= end_time:
            return False, f"Attendance window closed at 04:00 PM. Current time is {now.strftime('%I:%M %p')}."

        return True, f"Attendance window is OPEN (10:00 AM - 04:00 PM) on {day_name}."

    def get_active_slot(self, dt: Optional[datetime] = None) -> Optional[Dict]:
        """
        Determines which course/subject lecture is active right now based on weekly timetable.
        Lecture intervals are strictly start-inclusive and end-exclusive (start <= t < end).
        Returns the active slot dict with course metadata, or None if no lecture is scheduled.
        """
        now = dt or datetime.now()
        day_name = now.strftime("%A")

        if day_name not in self.operating_days:
            return None

        current_time = now.time()
        day_slots = WEEKLY_SCHEDULE.get(day_name, [])

        for slot in day_slots:
            s_h, s_m = map(int, slot["start"].split(":"))
            e_h, e_m = map(int, slot["end"].split(":"))
            start_t = dtime(s_h, s_m, 0)
            end_t = dtime(e_h, e_m, 0)

            # Start-inclusive, end-exclusive comparison: start <= current_time < end
            if start_t <= current_time < end_t:
                course = self.get_course_details(slot["course_code"])
                return {
                    "day": day_name,
                    "slot_id": slot["slot_id"],
                    "start": slot["start"],
                    "end": slot["end"],
                    "session_type": slot["session_type"],
                    "course_code": course["course_code"],
                    "course_name": course["course_name"],
                    "short_name": course["short_name"],
                    "faculty": course["faculty"],
                    "room": course["room"],
                }

        return None

    def get_schedule_status(self, dt: Optional[datetime] = None) -> Dict:
        """
        Returns comprehensive status of current timetable slot and operating window.
        Provides user-friendly messages for active lecture, breaks, gaps, or outside hours.
        """
        now = dt or datetime.now()
        day_name = now.strftime("%A")
        time_str = now.strftime("%I:%M %p")
        active = self.get_active_slot(now)

        if active:
            return {
                "is_active": True,
                "active_slot": active,
                "day": day_name,
                "time": time_str,
                "message": f"Active Lecture: {active['course_code']} - {active['course_name']} ({active['start']} - {active['end']})",
            }

        # Slot is inactive. Provide specific diagnostic reason.
        if day_name not in self.operating_days:
            reason = f"Attendance unavailable on {day_name} (Non-working day)."
        else:
            current_t = now.time()
            if current_t < dtime(self.start_hour, 0):
                reason = f"Outside class hours (Starts at 10:00 AM). Current time: {time_str}."
            elif current_t >= dtime(self.end_hour, 0):
                reason = f"Class hours ended at 04:00 PM. Current time: {time_str}."
            elif dtime(12, 15) <= current_t < dtime(13, 0) and day_name != "Saturday":
                reason = "Lunch Break (12:15 PM - 01:00 PM). No lecture scheduled."
            elif dtime(13, 15) <= current_t < dtime(14, 0) and day_name == "Saturday":
                reason = "Saturday Lunch / Break (01:15 PM - 02:00 PM). No lecture scheduled."
            else:
                reason = "Class Transit / Recess Interval. No lecture in session."

        return {
            "is_active": False,
            "active_slot": None,
            "day": day_name,
            "time": time_str,
            "message": f"NO ACTIVE LECTURE — ATTENDANCE DISABLED ({reason})",
        }

    def get_timetable_for_day(self, day_name: str) -> List[Dict]:
        """Returns all scheduled slots for a given day with course metadata enriched."""
        slots = WEEKLY_SCHEDULE.get(day_name, [])
        enriched = []
        for s in slots:
            c = self.get_course_details(s["course_code"])
            enriched.append({**s, **c})
        return enriched

    def get_all_courses(self) -> List[Dict]:
        """Returns list of all curriculum courses."""
        return CURRICULUM_COURSES
