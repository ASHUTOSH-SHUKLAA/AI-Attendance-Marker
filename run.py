"""
Unified CLI Entry Point for AI-Powered Classroom Attendance System.
Usage:
  python run.py --mode gui             (Launch Graphical User Interface)
  python run.py --mode live            (Start Real-Time Webcam Attendance Scanner)
  python run.py --mode scan --image P  (Scan a single classroom image file)
  python run.py --mode enroll          (Enroll a new student via webcam)
  python run.py --mode evaluate        (Run Accuracy, FAR, FRR, and FPS Benchmark)
  python run.py --mode report          (View Subject-Wise Attendance Review)
  python run.py --mode classes         (View Conducted Classes: Headcount & Attendee Roster)
  python run.py --mode timetable       (View Weekly Timetable Mon-Sat 10 AM - 4 PM)
  python run.py --mode clear-data      (Clear all stored data for clean real enrollment)
"""
import argparse
import sys
from pathlib import Path

import config
from src.database import DatabaseManager
from src.face_engine import FaceEngine
from src.attendance_system import AttendanceSystem
from src.enrollment import EnrollmentManager
from src.evaluate import AttendanceEvaluator


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI-Powered Classroom Attendance System - Semester 5 Computer Vision Project"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gui",
        choices=["gui", "live", "scan", "enroll", "evaluate", "rebuild", "report", "classes", "timetable", "clear-data"],
        help="Execution mode (default: 'gui')",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to classroom image for --mode scan",
    )
    parser.add_argument(
        "--course",
        type=str,
        default="AUTO",
        help="Course code (e.g. CS301, CS302, or 'AUTO' to use timetable slot)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=config.DEFAULT_SESSION_NAME,
        help="Session identifier (e.g. Lecture_01)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--liveness",
        action="store_true",
        help="Enable anti-spoofing / liveness texture verification",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Ensure face detection model is downloaded locally in models/
    from download_models import download_cascade
    download_cascade()

    db = DatabaseManager()
    engine = FaceEngine()

    if args.mode == "gui":
        print("[Launcher] Starting Desktop GUI Dashboard...")
        from src.gui_app import launch_gui
        launch_gui()

    elif args.mode == "live":
        print(f"[Launcher] Starting Real-Time Camera Scanner on camera index {args.camera}...")
        attendance_sys = AttendanceSystem(
            db=db,
            engine=engine,
            course_code=args.course,
            session_name=args.session,
            enable_liveness=args.liveness,
        )
        attendance_sys.run_live(camera_index=args.camera)

    elif args.mode == "scan":
        target_path = Path(args.image) if args.image else (config.SAMPLE_DIR / "classroom_lecture.jpg")
        if not target_path.exists():
            print(f"[Error] Target image does not exist: {target_path}")
            sys.exit(1)

        print(f"[Launcher] Scanning classroom image: {target_path}")
        attendance_sys = AttendanceSystem(
            db=db,
            engine=engine,
            course_code=args.course,
            session_name=args.session,
        )
        res = attendance_sys.scan_image(target_path)
        print(f" -> Total Faces Detected: {res['total_faces_detected']}")
        print(f" -> Marked Students: {len(res['marked_students'])}")
        for s in res['marked_students']:
            print(f"    - {s['name']} ({s['student_id']}) [Confidence: {s['confidence']}%]")
        print(f" -> Annotated Image Saved: {res['annotated_image']}")

    elif args.mode == "enroll":
        print("\n--- Enroll New Student ---")
        student_id = input("Enter Student ID (e.g. STU011): ").strip()
        name = input("Enter Full Name: ").strip()
        roll_no = input("Enter Roll Number: ").strip()
        branch = input("Enter Branch (default CSE): ").strip() or "CSE"

        enrollment_mgr = EnrollmentManager(db=db, engine=engine)
        success = enrollment_mgr.enroll_from_webcam(
            student_id=student_id,
            name=name,
            roll_no=roll_no,
            branch=branch,
            camera_index=args.camera,
        )
        if success:
            print(f"[Success] Enrolled {name} into system.")
        else:
            print("[Failed] Enrollment could not be completed.")

    elif args.mode == "evaluate":
        print("[Launcher] Running Biometric Accuracy & Performance Benchmark...")
        evaluator = AttendanceEvaluator(engine=engine)
        evaluator.run_full_evaluation(num_test_trials=100)

    elif args.mode == "clear-data":
        print("[Launcher] Clearing all stored students, encodings, and attendance logs...")
        from src.database import reset_all_data
        reset_all_data()
        print("[Done] All hardcoded and previous data cleared. Platform is ready for real student enrollment.")

    elif args.mode == "rebuild":
        print("[Launcher] Rebuilding face encodings and database from 'data/students/'...")
        enrollment_mgr = EnrollmentManager(db=db, engine=engine)
        count = enrollment_mgr.scan_and_rebuild_from_dataset()
        print(f"[Done] Rebuilt {count} students into database and encodings file.")

    elif args.mode == "report":
        print("\n=========================================================================================")
        print("          CURRICULUM ATTENDANCE REVIEW REPORT (<75% SHORTAGE WARNINGS)")
        print("=========================================================================================")
        summary = db.get_subject_wise_summary()
        header = f"{'Roll No':<10} {'Name':<18} {'CS301':<9} {'CS302':<9} {'CS303':<9} {'CS304':<9} {'CS305':<9} {'Overall':<9} {'Status':<12}"
        print(header)
        print("-" * len(header))
        for r in summary:
            status = "DEFICIT (<75%)" if r["is_defaulter"] else "ELIGIBLE"
            cs301 = f"{r['subjects'].get('CS301', {}).get('pct', 0.0)}%"
            cs302 = f"{r['subjects'].get('CS302', {}).get('pct', 0.0)}%"
            cs303 = f"{r['subjects'].get('CS303', {}).get('pct', 0.0)}%"
            cs304 = f"{r['subjects'].get('CS304', {}).get('pct', 0.0)}%"
            cs305 = f"{r['subjects'].get('CS305', {}).get('pct', 0.0)}%"
            overall = f"{r['overall_pct']}%"
            print(f"{r['roll_no']:<10} {r['name']:<18} {cs301:<9} {cs302:<9} {cs303:<9} {cs304:<9} {cs305:<9} {overall:<9} {status:<12}")
        print("=========================================================================================")
        csv_file = db.export_subject_wise_summary_csv()
        print(f"[Export] Report saved to: {csv_file}")

    elif args.mode == "classes":
        print("\n=========================================================================================")
        print("          CONDUCTED CLASS ATTENDANCE SESSIONS (WHO & HOW MANY WERE PRESENT)")
        print("=========================================================================================")
        sessions = db.get_class_attendance_summary()
        if not sessions:
            print("No class attendance sessions recorded yet.")
        else:
            for s in sessions:
                print(f"\n>> CLASS SESSION: {s['course_code']} - {s['course_name']} ({s['session_name']})")
                print(f"   Date: {s['date']} ({s['day_of_week']})")
                print(f"   Headcount: {s['total_present']} / {s['total_enrolled']} Present ({s['attendance_percentage']}%) | Absent: {s['total_absent']}")
                print("   -------------------------------------------------------------------------------------")
                print(f"   [PRESENT ATTENDEES ({s['total_present']})]:")
                if s["present_students"]:
                    for idx, p in enumerate(s["present_students"], 1):
                        print(f"     {idx:2d}. [{p['roll_no']}] {p['name']:<18} (ID: {p['student_id']}) | Time: {p['time']} | Conf: {p['confidence']}%")
                else:
                    print("     (None)")
                print(f"   [ABSENT STUDENTS ({s['total_absent']})]:")
                if s["absent_students"]:
                    for idx, a in enumerate(s["absent_students"], 1):
                        print(f"     {idx:2d}. [{a['roll_no']}] {a['name']:<18} (ID: {a['student_id']})")
                else:
                    print("     (All enrolled students attended)")
                print("   " + "-" * 85)
        print("=========================================================================================\n")
        csv_file = db.export_all_class_sessions_csv()
        print(f"[Export] Consolidated class headcount & roster summary exported to:\n  -> {csv_file}")

    elif args.mode == "timetable":
        from src.schedule_manager import ScheduleManager, WEEKLY_SCHEDULE
        sm = ScheduleManager()
        active = sm.get_active_slot()
        status = sm.get_schedule_status()
        print("\n=========================================================================================")
        print("       5TH SEMESTER WEEKLY TIMETABLE (OPERATIONAL WINDOW: MON-SAT, 10 AM - 4 PM)")
        print("=========================================================================================")
        if active:
            print(f"[*] CURRENT STATUS: ACTIVE LECTURE -> {active['course_code']} - {active['course_name']} ({active['start']} - {active['end']}) | Room: {active['room']}")
        else:
            print(f"[*] CURRENT STATUS: {status['message']}")
        for day, slots in WEEKLY_SCHEDULE.items():
            print(f"\n>> {day.upper()}")
            for s in slots:
                c = sm.get_course_details(s['course_code'])
                print(f"   [{s['start']} - {s['end']}] {s['course_code']}: {c['course_name']} ({s['session_type']}) | {c['faculty']} | {c['room']}")
        print("=========================================================================================\n")


if __name__ == "__main__":
    main()
