"""
Comprehensive Desktop Graphical User Interface (GUI) for AI Attendance System.
Built with standard Tkinter.
Features:
- Real-time camera scanner with subject-wise bounding box badges.
- Academic schedule enforcement (Monday-Saturday, 10:00 AM - 04:00 PM).
- Dynamic course/subject selector (Auto Timetable or manual course override).
- Subject-wise and curriculum-wide review dashboard (<75% defaulter warnings).
- Weekly academic timetable explorer.
- Quick webcam enrollment and accuracy benchmarking.
"""
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

import config
from src.database import DatabaseManager
from src.face_engine import FaceEngine
from src.attendance_system import AttendanceSystem
from src.enrollment import EnrollmentManager
from src.evaluate import AttendanceEvaluator
from src.schedule_manager import ScheduleManager, CURRICULUM_COURSES


class AttendanceGUI:
    """Desktop Dashboard for classroom attendance management with schedule integration."""

    def __init__(self, root: tk.Tk, demo_mode: bool = False):
        self.root = root
        self.root.title("AI-Powered Classroom Attendance System - CS301 / Semester 5")
        self.root.geometry("1240x800")
        self.root.minsize(1040, 680)
        self.initial_demo_mode = demo_mode

        # Apply dark modern theme
        self.bg_color = "#18181b"
        self.card_bg = "#27272a"
        self.accent_color = "#3b82f6"
        self.accent_green = "#10b981"
        self.accent_warn = "#f59e0b"
        self.accent_red = "#ef4444"
        self.text_color = "#f4f4f5"
        self.root.configure(bg=self.bg_color)

        # Core Backend instances
        self.db = DatabaseManager()
        self.engine = FaceEngine()
        self.schedule_mgr = ScheduleManager()
        self.attendance_sys = AttendanceSystem(db=self.db, engine=self.engine, demo_mode=demo_mode)
        self.enrollment_mgr = EnrollmentManager(db=self.db, engine=self.engine)

        # Camera & State
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None

        self._build_ui()
        self._load_registered_students()
        self._load_todays_attendance()
        self._load_class_sessions()
        self._load_subject_review()
        self._load_timetable_view("Monday")
        self._start_clock_ticker()

    def _build_ui(self):
        # 1. Header Banner
        header = tk.Frame(self.root, bg="#09090b", height=65)
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(
            header,
            text="🎓 AI-POWERED ATTENDANCE SYSTEM | 5th SEMESTER",
            font=("Segoe UI", 15, "bold"),
            fg="#60a5fa",
            bg="#09090b",
            padx=18,
            pady=12,
        )
        title.pack(side=tk.LEFT)

        self.lbl_clock = tk.Label(
            header,
            text="Loading Schedule...",
            font=("Segoe UI", 10),
            fg="#a1a1aa",
            bg="#09090b",
            padx=18,
        )
        self.lbl_clock.pack(side=tk.RIGHT)

        # 2. Timetable Active Lecture Status Banner & Demo Mode Controls
        toolbar = tk.Frame(self.root, bg="#18181b", padx=15, pady=8)
        toolbar.pack(fill=tk.X)

        self.lbl_active_lecture = tk.Label(
            toolbar,
            text="CHECKING ACTIVE LECTURE...",
            font=("Segoe UI", 10, "bold"),
            fg="#e4e4e7",
            bg="#27272a",
            padx=12,
            pady=4,
        )
        self.lbl_active_lecture.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # Demo Mode Toggle Checkbox
        self.var_demo_mode = tk.BooleanVar(value=self.initial_demo_mode)
        self.chk_demo = tk.Checkbutton(
            toolbar,
            text="🧪 Demo Mode (Bypass Schedule)",
            variable=self.var_demo_mode,
            command=self._on_toggle_demo_mode,
            font=("Segoe UI", 9, "bold"),
            fg="#fbbf24",
            bg="#18181b",
            selectcolor="#27272a",
            activebackground="#18181b",
            activeforeground="#fbbf24",
        )
        self.chk_demo.pack(side=tk.LEFT, padx=(0, 6))

        # Demo Subject Selector
        self.demo_course_options = [
            f"{c['course_code']} - {c['course_name']}" for c in CURRICULUM_COURSES
        ]
        self.var_demo_course = tk.StringVar(value=self.demo_course_options[0])
        self.cmb_demo_course = ttk.Combobox(
            toolbar,
            textvariable=self.var_demo_course,
            values=self.demo_course_options,
            state="readonly" if self.initial_demo_mode else "disabled",
            width=28,
            font=("Segoe UI", 9),
        )
        self.cmb_demo_course.pack(side=tk.LEFT, padx=(0, 10))
        self.cmb_demo_course.bind("<<ComboboxSelected>>", self._on_demo_course_selected)

        self.lbl_window_badge = tk.Label(
            toolbar,
            text="CHECKING STATUS",
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg="#27272a",
            padx=12,
            pady=4,
        )
        self.lbl_window_badge.pack(side=tk.RIGHT)

        # 3. Main Body Split: Left Video Feed, Right Management Notebook
        main_body = tk.Frame(self.root, bg=self.bg_color, padx=12, pady=10)
        main_body.pack(fill=tk.BOTH, expand=True)

        # Left Column: Video Feed & Controls
        left_col = tk.Frame(main_body, bg=self.card_bg, padx=10, pady=10, relief=tk.RIDGE, bd=1)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(
            left_col,
            text="Camera Scanner Inactive\nClick '▶ Start Camera Scanner' below",
            font=("Segoe UI", 12),
            fg="#a1a1aa",
            bg="#18181b",
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Camera Controls Row
        cam_btns = tk.Frame(left_col, bg=self.card_bg)
        cam_btns.pack(fill=tk.X)

        self.btn_toggle_cam = tk.Button(
            cam_btns,
            text="▶ Start Camera Scanner",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_green,
            fg="#ffffff",
            padx=14,
            pady=8,
            relief=tk.FLAT,
            command=self.toggle_camera,
        )
        self.btn_toggle_cam.pack(side=tk.LEFT, padx=4)

        self.btn_scan_image = tk.Button(
            cam_btns,
            text="📁 Scan Classroom Photo",
            font=("Segoe UI", 10),
            bg=self.accent_color,
            fg="#ffffff",
            padx=12,
            pady=8,
            relief=tk.FLAT,
            command=self.scan_classroom_file,
        )
        self.btn_scan_image.pack(side=tk.LEFT, padx=4)

        self.btn_export = tk.Button(
            cam_btns,
            text="💾 Export Daily CSV",
            font=("Segoe UI", 10),
            bg="#4b5563",
            fg="#ffffff",
            padx=12,
            pady=8,
            relief=tk.FLAT,
            command=self.export_csv,
        )
        self.btn_export.pack(side=tk.RIGHT, padx=4)

        # Right Column: Multi-tab Panel
        right_col = tk.Frame(main_body, bg=self.card_bg, width=540, padx=8, pady=8, relief=tk.RIDGE, bd=1)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(12, 0))

        self.notebook = ttk.Notebook(right_col)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Live Today's Attendance Log
        self.tab_attendance = tk.Frame(self.notebook, bg=self.card_bg, padx=6, pady=6)
        self.notebook.add(self.tab_attendance, text="📋 Today's Log")
        self._build_attendance_tab()

        # Tab 2: Class Attendance Roster (Who & How Many Were Present)
        self.tab_classes = tk.Frame(self.notebook, bg=self.card_bg, padx=6, pady=6)
        self.notebook.add(self.tab_classes, text="🏫 Class Roster")
        self._build_classes_tab()

        # Tab 3: Subject-Wise & Curriculum Review
        self.tab_review = tk.Frame(self.notebook, bg=self.card_bg, padx=6, pady=6)
        self.notebook.add(self.tab_review, text="📊 Subject Review (<75%)")
        self._build_review_tab()

        # Tab 3: Weekly Timetable
        self.tab_timetable = tk.Frame(self.notebook, bg=self.card_bg, padx=6, pady=6)
        self.notebook.add(self.tab_timetable, text="📅 Timetable (10-4)")
        self._build_timetable_tab()

        # Tab 4: Registered Students Roster
        self.tab_students = tk.Frame(self.notebook, bg=self.card_bg, padx=6, pady=6)
        self.notebook.add(self.tab_students, text="👥 Enrolled (10+)")
        self._build_students_tab()

        # Tab 5: Quick Enrollment
        self.tab_enroll = tk.Frame(self.notebook, bg=self.card_bg, padx=6, pady=6)
        self.notebook.add(self.tab_enroll, text="➕ Enroll")
        self._build_enrollment_tab()

        # Tab 6: Accuracy Benchmark
        self.tab_metrics = tk.Frame(self.notebook, bg=self.card_bg, padx=6, pady=6)
        self.notebook.add(self.tab_metrics, text="📈 Accuracy")
        self._build_metrics_tab()

        # Status Bar
        self.status_bar = tk.Label(
            self.root,
            text=f"Ready | Enrolled: {self.engine.total_registered} students | Operating Window: Mon-Sat 10:00 AM - 04:00 PM",
            font=("Segoe UI", 9),
            fg="#9ca3af",
            bg="#09090b",
            anchor="w",
            padx=15,
            pady=4,
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _start_clock_ticker(self):
        """Refreshes the header clock and active lecture timetable status every second."""
        now = datetime.now()
        day_str = now.strftime("%A, %b %d, %Y")
        time_str = now.strftime("%I:%M:%S %p")
        active = self.schedule_mgr.get_active_slot(now)
        status_info = self.schedule_mgr.get_schedule_status(now)

        self.lbl_clock.config(text=f"{day_str} | {time_str}")

        if self.var_demo_mode.get():
            c_code, c_name, _ = self.attendance_sys.resolve_current_course(dt=now)
            demo_text = f"🧪 DEMO MODE: {c_code} \u2014 {c_name} [SCHEDULE BYPASS ACTIVE]"
            self.lbl_active_lecture.config(
                text=demo_text,
                fg="#fef08a",
                bg="#713f12",
            )
            self.lbl_window_badge.config(
                text="🧪 DEMO (BYPASS ACTIVE)",
                bg="#854d0e",
                fg="#ffffff",
            )
            self.status_bar.config(
                text=f"Demo Mode: Schedule bypass active | Attendance enabled for {c_code} ({c_name}) anytime"
            )
        elif active:
            lecture_text = (
                f"ACTIVE LECTURE: {active['course_code']} \u2014 {active['course_name']} "
                f"({active['start']} - {active['end']}) | Room: {active['room']} | Faculty: {active['faculty']}"
            )
            self.lbl_active_lecture.config(
                text=lecture_text,
                fg="#34d399",
                bg="#14532d",
            )
            self.lbl_window_badge.config(
                text="● ATTENDANCE ACTIVE",
                bg="#15803d",
                fg="#ffffff",
            )
            self.status_bar.config(
                text=f"Active Lecture: {active['course_code']} ({active['course_name']}) | One attendance per student per lecture"
            )
        else:
            self.lbl_active_lecture.config(
                text=f"NO ACTIVE LECTURE \u2014 ATTENDANCE DISABLED [{status_info['message']}]",
                fg="#fca5a5",
                bg="#450a0a",
            )
            self.lbl_window_badge.config(
                text="○ ATTENDANCE DISABLED",
                bg="#7f1d1d",
                fg="#fecaca",
            )
            self.status_bar.config(
                text=f"No Active Lecture | {status_info['message']} | Mon-Sat 10:00 AM - 04:00 PM (Or enable '🧪 Demo Mode')"
            )

        self.root.after(1000, self._start_clock_ticker)

    def _on_toggle_demo_mode(self):
        is_demo = self.var_demo_mode.get()
        self.attendance_sys.demo_mode = is_demo
        if is_demo:
            self.cmb_demo_course.config(state="readonly")
            self._on_demo_course_selected()
            messagebox.showinfo(
                "Demo Mode Activated",
                "Timetable schedule enforcement is now BYPASSED.\n\n"
                "You can now test and mark attendance at any time for the selected subject.",
            )
        else:
            self.cmb_demo_course.config(state="disabled")
            self.attendance_sys.course_code = "AUTO"
            messagebox.showinfo(
                "Strict Schedule Restored",
                "Timetable schedule enforcement is now RESTORED.\n\n"
                "Attendance will only be marked during active scheduled lecture slots (Mon-Sat 10 AM - 4 PM).",
            )

    def _on_demo_course_selected(self, event=None):
        if not self.var_demo_mode.get():
            return
        val = self.var_demo_course.get()
        code = val.split(" - ")[0].strip()
        self.attendance_sys.course_code = code

    def _build_attendance_tab(self):
        top_row = tk.Frame(self.tab_attendance, bg=self.card_bg)
        top_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            top_row,
            text="Today's Recorded Attendance Log",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_color,
            bg=self.card_bg,
        ).pack(side=tk.LEFT)

        columns = ("id", "name", "course", "time", "conf")
        self.tree_attendance = ttk.Treeview(self.tab_attendance, columns=columns, show="headings", height=18)
        self.tree_attendance.heading("id", text="ID")
        self.tree_attendance.heading("name", text="Student Name")
        self.tree_attendance.heading("course", text="Course")
        self.tree_attendance.heading("time", text="Timestamp")
        self.tree_attendance.heading("conf", text="Confidence")

        self.tree_attendance.column("id", width=65)
        self.tree_attendance.column("name", width=125)
        self.tree_attendance.column("course", width=65)
        self.tree_attendance.column("time", width=80)
        self.tree_attendance.column("conf", width=70)
        self.tree_attendance.pack(fill=tk.BOTH, expand=True)

    def _build_classes_tab(self):
        # 1. Header with Refresh
        top_bar = tk.Frame(self.tab_classes, bg=self.card_bg)
        top_bar.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            top_bar,
            text="Conducted Classes (Select class to view who was present)",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_color,
            bg=self.card_bg,
        ).pack(side=tk.LEFT)

        tk.Button(
            top_bar,
            text="🔄 Refresh",
            font=("Segoe UI", 8),
            bg="#3f3f46",
            fg="#ffffff",
            padx=6,
            pady=1,
            relief=tk.FLAT,
            command=self._load_class_sessions,
        ).pack(side=tk.RIGHT)

        # 2. Table of conducted classes
        cols = ("date", "day", "code", "subject", "present", "enrolled", "pct")
        self.tree_classes = ttk.Treeview(self.tab_classes, columns=cols, show="headings", height=5)
        self.tree_classes.heading("date", text="Date")
        self.tree_classes.heading("day", text="Day")
        self.tree_classes.heading("code", text="Course")
        self.tree_classes.heading("subject", text="Subject Name")
        self.tree_classes.heading("present", text="Present")
        self.tree_classes.heading("enrolled", text="Enrolled")
        self.tree_classes.heading("pct", text="Attd %")

        self.tree_classes.column("date", width=75)
        self.tree_classes.column("day", width=70)
        self.tree_classes.column("code", width=55)
        self.tree_classes.column("subject", width=120)
        self.tree_classes.column("present", width=55)
        self.tree_classes.column("enrolled", width=55)
        self.tree_classes.column("pct", width=50)
        self.tree_classes.pack(fill=tk.X, pady=(0, 4))
        self.tree_classes.bind("<<TreeviewSelect>>", self._on_class_selected)

        # 3. Selected class details header & Export Button
        details_bar = tk.Frame(self.tab_classes, bg=self.card_bg)
        details_bar.pack(fill=tk.X, pady=(4, 4))

        self.lbl_roster_header = tk.Label(
            details_bar,
            text="Select a class session above to view attendee & absentee list",
            font=("Segoe UI", 9, "bold"),
            fg="#60a5fa",
            bg=self.card_bg,
        )
        self.lbl_roster_header.pack(side=tk.LEFT)

        self.btn_export_roster = tk.Button(
            details_bar,
            text="📥 Export Roster CSV",
            font=("Segoe UI", 8, "bold"),
            bg=self.accent_color,
            fg="#ffffff",
            padx=8,
            pady=2,
            relief=tk.FLAT,
            command=self._export_selected_class_roster,
        )
        self.btn_export_roster.pack(side=tk.RIGHT)

        # 4. Sub-notebook for Present vs Absent
        self.roster_notebook = ttk.Notebook(self.tab_classes)
        self.roster_notebook.pack(fill=tk.BOTH, expand=True)

        # Sub-tab A: Present Students (Who Was Present)
        self.subtab_present = tk.Frame(self.roster_notebook, bg=self.card_bg, padx=4, pady=4)
        self.roster_notebook.add(self.subtab_present, text="✅ Present Students")

        cols_p = ("roll", "id", "name", "time", "conf")
        self.tree_class_present = ttk.Treeview(self.subtab_present, columns=cols_p, show="headings", height=8)
        self.tree_class_present.heading("roll", text="Roll No")
        self.tree_class_present.heading("id", text="Student ID")
        self.tree_class_present.heading("name", text="Student Name")
        self.tree_class_present.heading("time", text="Time Marked")
        self.tree_class_present.heading("conf", text="Confidence")

        self.tree_class_present.column("roll", width=80)
        self.tree_class_present.column("id", width=70)
        self.tree_class_present.column("name", width=140)
        self.tree_class_present.column("time", width=85)
        self.tree_class_present.column("conf", width=75)
        self.tree_class_present.pack(fill=tk.BOTH, expand=True)

        # Sub-tab B: Absent Students (Who Was Absent)
        self.subtab_absent = tk.Frame(self.roster_notebook, bg=self.card_bg, padx=4, pady=4)
        self.roster_notebook.add(self.subtab_absent, text="❌ Absent Students")

        cols_a = ("roll", "id", "name", "branch")
        self.tree_class_absent = ttk.Treeview(self.subtab_absent, columns=cols_a, show="headings", height=8)
        self.tree_class_absent.heading("roll", text="Roll No")
        self.tree_class_absent.heading("id", text="Student ID")
        self.tree_class_absent.heading("name", text="Student Name")
        self.tree_class_absent.heading("branch", text="Branch")

        self.tree_class_absent.column("roll", width=80)
        self.tree_class_absent.column("id", width=75)
        self.tree_class_absent.column("name", width=140)
        self.tree_class_absent.column("branch", width=70)
        self.tree_class_absent.pack(fill=tk.BOTH, expand=True)

    def _build_review_tab(self):
        lbl = tk.Label(
            self.tab_review,
            text="Subject-Wise Attendance Review & Shortage (<75%) Alerts",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_color,
            bg=self.card_bg,
        )
        lbl.pack(anchor="w", pady=(0, 6))

        cols = ("roll", "name", "cs301", "cs302", "cs303", "cs304", "cs305", "overall", "status")
        self.tree_review = ttk.Treeview(self.tab_review, columns=cols, show="headings", height=15)
        self.tree_review.heading("roll", text="Roll No")
        self.tree_review.heading("name", text="Name")
        self.tree_review.heading("cs301", text="CS301 (CV)")
        self.tree_review.heading("cs302", text="CS302 (OS)")
        self.tree_review.heading("cs303", text="CS303 (DBMS)")
        self.tree_review.heading("cs304", text="CS304 (CN)")
        self.tree_review.heading("cs305", text="CS305 (SE)")
        self.tree_review.heading("overall", text="Overall %")
        self.tree_review.heading("status", text="Eligibility Status")

        self.tree_review.column("roll", width=75)
        self.tree_review.column("name", width=110)
        self.tree_review.column("cs301", width=65)
        self.tree_review.column("cs302", width=65)
        self.tree_review.column("cs303", width=70)
        self.tree_review.column("cs304", width=65)
        self.tree_review.column("cs305", width=65)
        self.tree_review.column("overall", width=65)
        self.tree_review.column("status", width=105)
        self.tree_review.pack(fill=tk.BOTH, expand=True)

        btn_row = tk.Frame(self.tab_review, bg=self.card_bg)
        btn_row.pack(fill=tk.X, pady=(8, 0))

        tk.Button(
            btn_row,
            text="🔄 Refresh Analytics",
            font=("Segoe UI", 9, "bold"),
            bg=self.accent_color,
            fg="#ffffff",
            padx=10,
            pady=5,
            relief=tk.FLAT,
            command=self._load_subject_review,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_row,
            text="💾 Export Review CSV",
            font=("Segoe UI", 9),
            bg="#4b5563",
            fg="#ffffff",
            padx=10,
            pady=5,
            relief=tk.FLAT,
            command=self._export_review_csv,
        ).pack(side=tk.RIGHT, padx=4)

    def _build_timetable_tab(self):
        day_row = tk.Frame(self.tab_timetable, bg=self.card_bg)
        day_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            day_row,
            text="Select Day (Mon-Sat, 10 AM - 4 PM):",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_color,
            bg=self.card_bg,
        ).pack(side=tk.LEFT)

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        for d in days:
            btn = tk.Button(
                day_row,
                text=d[:3],
                font=("Segoe UI", 8),
                bg="#3f3f46",
                fg="#ffffff",
                padx=6,
                pady=2,
                relief=tk.FLAT,
                command=lambda day=d: self._load_timetable_view(day),
            )
            btn.pack(side=tk.LEFT, padx=2)

        self.lbl_selected_day = tk.Label(
            self.tab_timetable,
            text="Timetable for Monday (10:00 AM - 04:00 PM)",
            font=("Segoe UI", 10, "bold"),
            fg="#60a5fa",
            bg=self.card_bg,
        )
        self.lbl_selected_day.pack(anchor="w", pady=(0, 4))

        cols = ("time", "code", "subject", "type", "faculty", "room")
        self.tree_tt = ttk.Treeview(self.tab_timetable, columns=cols, show="headings", height=15)
        self.tree_tt.heading("time", text="Time Slot")
        self.tree_tt.heading("code", text="Course")
        self.tree_tt.heading("subject", text="Subject Name")
        self.tree_tt.heading("type", text="Session Type")
        self.tree_tt.heading("faculty", text="Faculty")
        self.tree_tt.heading("room", text="Room")

        self.tree_tt.column("time", width=90)
        self.tree_tt.column("code", width=60)
        self.tree_tt.column("subject", width=140)
        self.tree_tt.column("type", width=90)
        self.tree_tt.column("faculty", width=110)
        self.tree_tt.column("room", width=70)
        self.tree_tt.pack(fill=tk.BOTH, expand=True)

    def _build_students_tab(self):
        lbl = tk.Label(
            self.tab_students,
            text="Registered Students Directory (Only Real Enrolled Students)",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_color,
            bg=self.card_bg,
        )
        lbl.pack(anchor="w", pady=(0, 6))

        cols = ("id", "name", "roll", "branch")
        self.tree_students = ttk.Treeview(self.tab_students, columns=cols, show="headings", height=16)
        self.tree_students.heading("id", text="ID")
        self.tree_students.heading("name", text="Name")
        self.tree_students.heading("roll", text="Roll No")
        self.tree_students.heading("branch", text="Branch")

        self.tree_students.column("id", width=60)
        self.tree_students.column("name", width=140)
        self.tree_students.column("roll", width=90)
        self.tree_students.column("branch", width=60)
        self.tree_students.pack(fill=tk.BOTH, expand=True)

        btn_row = tk.Frame(self.tab_students, bg=self.card_bg)
        btn_row.pack(fill=tk.X, pady=(6, 0))

        tk.Button(
            btn_row,
            text="🗑 Clear All Stored Data",
            font=("Segoe UI", 9, "bold"),
            bg="#991b1b",
            fg="#ffffff",
            padx=10,
            pady=4,
            relief=tk.FLAT,
            command=self.reset_platform_data,
        ).pack(side=tk.RIGHT)

    def _build_enrollment_tab(self):
        form = tk.Frame(self.tab_enroll, bg=self.card_bg)
        form.pack(fill=tk.X, pady=10)

        def make_entry(parent, label_text):
            f = tk.Frame(parent, bg=self.card_bg)
            f.pack(fill=tk.X, pady=4)
            lbl = tk.Label(f, text=label_text, width=12, anchor="w", fg=self.text_color, bg=self.card_bg)
            lbl.pack(side=tk.LEFT)
            ent = tk.Entry(f, font=("Segoe UI", 10))
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            return ent

        self.ent_id = make_entry(form, "Student ID:")
        self.ent_name = make_entry(form, "Full Name:")
        self.ent_roll = make_entry(form, "Roll Number:")
        self.ent_branch = make_entry(form, "Branch:")

        self.ent_id.insert(0, f"STU{self.engine.total_registered + 1:03d}")
        self.ent_branch.insert(0, "CSE")

        btn_cam_enroll = tk.Button(
            self.tab_enroll,
            text="📸 Enroll via Webcam (5 Captures)",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_color,
            fg="#ffffff",
            pady=8,
            relief=tk.FLAT,
            command=self.start_webcam_enrollment,
        )
        btn_cam_enroll.pack(fill=tk.X, pady=(15, 6))

    def _build_metrics_tab(self):
        lbl = tk.Label(
            self.tab_metrics,
            text="Performance Evaluation Benchmark",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_color,
            bg=self.card_bg,
        )
        lbl.pack(anchor="w", pady=(0, 6))

        self.txt_metrics = tk.Text(
            self.tab_metrics,
            height=18,
            font=("Consolas", 9),
            bg="#18181b",
            fg="#34d399",
            padx=8,
            pady=8,
        )
        self.txt_metrics.pack(fill=tk.BOTH, expand=True)

        btn_eval = tk.Button(
            self.tab_metrics,
            text="🚀 Run Accuracy & Metrics Benchmark",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_green,
            fg="#ffffff",
            pady=6,
            relief=tk.FLAT,
            command=self.run_benchmark,
        )
        btn_eval.pack(fill=tk.X, pady=(8, 0))

    def _load_registered_students(self):
        for item in self.tree_students.get_children():
            self.tree_students.delete(item)
        students = self.db.get_all_students()
        for s in students:
            self.tree_students.insert("", tk.END, values=(s["student_id"], s["name"], s["roll_no"], s["branch"]))

    def _load_todays_attendance(self):
        for item in self.tree_attendance.get_children():
            self.tree_attendance.delete(item)
        rows = self.db.get_attendance_for_date()
        for r in rows:
            conf_str = f"{r['confidence']:.1f}%" if r["confidence"] <= 100 else f"{r['confidence']:.1f}"
            course_lbl = r.get("course_code", "CS301")
            self.tree_attendance.insert(
                "",
                tk.END,
                values=(r["student_id"], r["name"], course_lbl, r["time"], conf_str),
            )
        self._load_class_sessions()

    def _load_class_sessions(self):
        for item in self.tree_classes.get_children():
            self.tree_classes.delete(item)

        sessions = self.db.get_class_attendance_summary()
        for s in sessions:
            self.tree_classes.insert(
                "",
                tk.END,
                values=(
                    s["date"],
                    s["day_of_week"][:3],
                    s["course_code"],
                    s["course_name"],
                    s["total_present"],
                    s["total_enrolled"],
                    f"{s['attendance_percentage']}%",
                ),
            )

        children = self.tree_classes.get_children()
        if children:
            self.tree_classes.selection_set(children[0])
            self._on_class_selected()
        else:
            self.lbl_roster_header.config(text="No class sessions recorded yet.")
            for item in self.tree_class_present.get_children():
                self.tree_class_present.delete(item)
            for item in self.tree_class_absent.get_children():
                self.tree_class_absent.delete(item)

    def _on_class_selected(self, event=None):
        selected = self.tree_classes.selection()
        if not selected:
            return

        values = self.tree_classes.item(selected[0], "values")
        if not values:
            return

        date_str = values[0]
        course_code = values[2]
        roster = self.db.get_class_roster_for_session(date_str, course_code)
        if not roster:
            return

        header_text = (
            f"Roster for {roster['course_code']} ({roster['course_name']}) on {roster['date']} | "
            f"Headcount: {roster['total_present']} / {roster['total_enrolled']} ({roster['attendance_percentage']}%) | "
            f"Absent: {roster['total_absent']}"
        )
        self.lbl_roster_header.config(text=header_text)
        self.roster_notebook.tab(0, text=f"✅ Present ({roster['total_present']})")
        self.roster_notebook.tab(1, text=f"❌ Absent ({roster['total_absent']})")

        # Populate Present Students
        for item in self.tree_class_present.get_children():
            self.tree_class_present.delete(item)
        for p in roster["present_students"]:
            conf_str = f"{p['confidence']:.1f}%" if p["confidence"] <= 100 else f"{p['confidence']:.1f}"
            self.tree_class_present.insert(
                "",
                tk.END,
                values=(p["roll_no"], p["student_id"], p["name"], p["time"], conf_str),
            )

        # Populate Absent Students
        for item in self.tree_class_absent.get_children():
            self.tree_class_absent.delete(item)
        for a in roster["absent_students"]:
            self.tree_class_absent.insert(
                "",
                tk.END,
                values=(a["roll_no"], a["student_id"], a["name"], a["branch"]),
            )

    def _export_selected_class_roster(self):
        selected = self.tree_classes.selection()
        if not selected:
            messagebox.showinfo("Export Roster", "Please select a class session to export.")
            return

        values = self.tree_classes.item(selected[0], "values")
        date_str = values[0]
        course_code = values[2]
        csv_file = self.db.export_class_roster_csv(date_str, course_code)
        messagebox.showinfo(
            "Export Successful",
            f"Class attendance roster exported to:\n{csv_file.resolve()}",
        )

    def _load_subject_review(self):
        for item in self.tree_review.get_children():
            self.tree_review.delete(item)

        summary = self.db.get_subject_wise_summary()
        for row in summary:
            status_text = "SHORTAGE (<75%)" if row["is_defaulter"] else "ELIGIBLE"
            tag = "defaulter" if row["is_defaulter"] else "eligible"

            self.tree_review.insert(
                "",
                tk.END,
                values=(
                    row["roll_no"],
                    row["name"],
                    f"{row['subjects'].get('CS301', {}).get('pct', 0.0)}%",
                    f"{row['subjects'].get('CS302', {}).get('pct', 0.0)}%",
                    f"{row['subjects'].get('CS303', {}).get('pct', 0.0)}%",
                    f"{row['subjects'].get('CS304', {}).get('pct', 0.0)}%",
                    f"{row['subjects'].get('CS305', {}).get('pct', 0.0)}%",
                    f"{row['overall_pct']}%",
                    status_text,
                ),
                tags=(tag,),
            )

        self.tree_review.tag_configure("defaulter", foreground="#f87171")
        self.tree_review.tag_configure("eligible", foreground="#34d399")

    def _load_timetable_view(self, day_name: str):
        self.lbl_selected_day.config(text=f"Timetable for {day_name} (Operational Window: 10:00 AM - 04:00 PM)")
        for item in self.tree_tt.get_children():
            self.tree_tt.delete(item)

        slots = self.schedule_mgr.get_timetable_for_day(day_name)
        for s in slots:
            time_slot = f"{s['start']} - {s['end']}"
            self.tree_tt.insert(
                "",
                tk.END,
                values=(time_slot, s["course_code"], s["course_name"], s["session_type"], s["faculty"], s["room"]),
            )

    def _export_review_csv(self):
        csv_file = self.db.export_subject_wise_summary_csv()
        messagebox.showinfo("Export Successful", f"Subject-Wise Review exported to:\n{csv_file.resolve()}")

    def toggle_camera(self):
        if self.is_camera_running:
            self.is_camera_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.btn_toggle_cam.config(text="▶ Start Camera Scanner", bg=self.accent_green)
            self.video_label.config(image="", text="Camera Feed Stopped.")
            self.status_bar.config(text="Camera Stopped.")
        else:
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", f"Unable to open camera at index {config.CAMERA_INDEX}.")
                return
            self.is_camera_running = True
            self.btn_toggle_cam.config(text="⏹ Stop Camera", bg=self.accent_red)
            self.status_bar.config(text="Scanning live camera feed for subject attendance...")
            self._update_camera_stream()

    def _update_camera_stream(self):
        if not self.is_camera_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            # Process face detection every 2 frames
            faces = self.engine.process_frame(frame, scale=0.5)
            for face in faces:
                if face.is_known:
                    self.attendance_sys.trigger_attendance(face)

            display = self.attendance_sys.draw_hud(frame, faces, fps=28.0)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((680, 420), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk, text="")

            self._load_todays_attendance()

        self.root.after(30, self._update_camera_stream)

    def scan_classroom_file(self):
        path = filedialog.askopenfilename(
            title="Select Classroom Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return

        result = self.attendance_sys.scan_image(Path(path))
        annotated_path = result["annotated_image"]

        # Display result image
        img = Image.open(str(annotated_path))
        img = img.resize((680, 420), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk, text="")

        self._load_todays_attendance()
        self._load_subject_review()
        messagebox.showinfo(
            "Scan Complete",
            f"Detected {result['total_faces_detected']} faces.\n"
            f"Marked attendance for {len(result['marked_students'])} registered students.\n"
            f"Saved annotated image to {annotated_path.name}",
        )

    def export_csv(self):
        csv_file = self.db.export_to_csv()
        messagebox.showinfo("Export Successful", f"Daily attendance log exported to:\n{csv_file.resolve()}")

    def start_webcam_enrollment(self):
        s_id = self.ent_id.get().strip()
        name = self.ent_name.get().strip()
        roll = self.ent_roll.get().strip()
        branch = self.ent_branch.get().strip()

        if not s_id or not name or not roll:
            messagebox.showwarning("Incomplete Form", "Please fill Student ID, Name, and Roll Number.")
            return

        if self.is_camera_running:
            self.toggle_camera()

        def run_enroll():
            success = self.enrollment_mgr.enroll_from_webcam(
                student_id=s_id,
                name=name,
                roll_no=roll,
                branch=branch,
            )
            self.root.after(100, lambda: self._on_enrollment_complete(success, name))

        threading.Thread(target=run_enroll, daemon=True).start()

    def _on_enrollment_complete(self, success: bool, name: str):
        if success:
            messagebox.showinfo("Enrollment Success", f"{name} successfully enrolled!")
            self._load_registered_students()
            self._load_subject_review()
            self.status_bar.config(text=f"Total Enrolled Students: {self.engine.total_registered}")
        else:
            messagebox.showerror("Enrollment Failed", "Could not complete enrollment.")

    def reset_platform_data(self):
        confirm = messagebox.askyesno(
            "Clear All Stored Data",
            "Are you sure you want to clear all stored students, facial vectors, and attendance records?\n\n"
            "This ensures ONLY real students you enroll will be stored on the platform.",
            icon="warning",
        )
        if not confirm:
            return

        from src.database import reset_all_data
        reset_all_data()

        # Reload components
        self.engine.load_encodings()
        self._load_registered_students()
        self._load_todays_attendance()
        self._load_subject_review()
        self.status_bar.config(text="Platform Reset Complete: 0 students registered. Ready for real student enrollment.")
        messagebox.showinfo("Success", "All previous data cleared successfully!\nYou can now enroll real students in the '➕ Enroll' tab.")

    def run_benchmark(self):
        evaluator = AttendanceEvaluator(engine=self.engine)
        results = evaluator.run_full_evaluation(num_test_trials=100)
        summary_txt = evaluator.format_text_summary(results)
        self.txt_metrics.delete("1.0", tk.END)
        self.txt_metrics.insert(tk.END, summary_txt)


def launch_gui(demo_mode: bool = False):
    root = tk.Tk()
    app = AttendanceGUI(root, demo_mode=demo_mode)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
