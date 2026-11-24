# 🎓 AI-Powered Classroom Attendance System
### Computer Vision Project Assignment (Semester 5 - CS301)

A real-time, deep-learning powered facial recognition attendance system built using **OpenCV**, **Python**, **Deep Residual Metric Learning (128D Embeddings)**, **Pandas**, and **SQLite**.

Now fully equipped with **Subject-Wise Curriculum Tracking**, **Timetable Schedule Enforcement (Monday–Saturday, 10:00 AM – 04:00 PM)**, and a **Subject Attendance Review Dashboard (<75% Defaulter Warning)**.

---

## 📌 Project Overview & Deliverables Checklist

- [x] **Working Prototype with Webcam Feed:** Real-time multi-face detection, green/red bounding boxes, name badges, confidence scores, and HUD telemetry.
- [x] **Authentic Student Registration Pipeline:** Supports registering students (e.g. 10+ students) via live webcam capture or photo enrollment, storing facial vectors and profiles dynamically with zero hardcoded data.
- [x] **Subject-Wise & Curriculum-Wise Tracking:** Enforces attendance tagging per subject (`CS301` Computer Vision, `CS302` OS, `CS303` DBMS, `CS304` CN, `CS305` SE & Lab).
- [x] **Academic Timetable & Schedule Enforcement:** Built-in weekly timetable from **Monday to Saturday (10:00 AM – 04:00 PM)**. Real-time slot resolution with automated window check.
- [x] **Daily Attendance Policy (One Mark Per Subject Per Day):** Enforces a database constraint `UNIQUE(student_id, date, course_code)`—students can mark attendance for each lecture session once per day without duplicate entries.
- [x] **Subject Review & Defaulter Dashboard (<75% Criteria):** Computes total lectures conducted vs attended per subject and flags students below university eligibility.
- [x] **Automated Attendance Log Generation:** Automatically logs attendance into SQLite database (`data/attendance.db`) and timestamped CSV files (`data/attendance_logs/`).
- [x] **Evaluation of System Accuracy and Limitations:** Quantitative benchmarking reporting **97.0% Accuracy**, **2.0% FAR**, **4.0% FRR**, and boundary analysis across lighting, noise, and occlusions.
- [x] **Anti-Spoofing / Liveness Check:** Eye-Blink detection (Eye Aspect Ratio - EAR) and Laplacian texture blur filtering to prevent photo/screen replay attacks.
- [x] **Modern Desktop Graphical User Interface (GUI):** Built with Tkinter with live video, course selector, timetable viewer, and review analytics.
- [x] **Complete Mid-Term Evaluation Report:** Formatted markdown report located at [`reports/MID_TERM_EVALUATION_REPORT.md`](file:///c:/Users/Ashuu/Documents/Sem%205/Projects/CV%20Assignment/reports/MID_TERM_EVALUATION_REPORT.md).

---

## 📁 Repository Structure

```text
CV Assignment/
│
├── run.py                          # Unified CLI entry point for all modes
├── config.py                       # Global parameters, schedule windows, and paths
├── download_models.py              # Auto-downloader for local OpenCV Haar Cascade
├── requirements.txt                # Pinned Python package dependencies
├── README.md                       # Documentation & execution manual
│
├── src/                            # Source code modules
│   ├── __init__.py                 # Package initializer
│   ├── schedule_manager.py         # 5th Semester Curriculum Timetable & Operating Window (Mon-Sat 10-4)
│   ├── face_engine.py              # Face detection, 128D embedding extraction & matching
│   ├── database.py                 # SQLite storage, subject tracking, anti-duplicate & review reports
│   ├── attendance_system.py        # Real-time camera feed processor and schedule HUD renderer
│   ├── enrollment.py               # Live webcam and batch image student enrollment tool
│   ├── liveness.py                 # Anti-spoofing module (EAR blink & texture variance)
│   ├── evaluate.py                 # Accuracy, confusion matrix, FAR/FRR & latency evaluator
│   ├── gui_app.py                  # Tkinter-based interactive desktop application
│   └── sample_data_generator.py    # Generator for 10-student cohort & classroom scenes
│
├── data/                           # Data storage directory
│   ├── students/                   # Directory containing enrolled student portrait folders
│   ├── encodings.pkl               # Serialized face vectors gallery
│   ├── attendance.db               # SQLite database (students, courses & attendance tables)
│   └── attendance_logs/            # Generated daily & curriculum review CSV records
│       ├── attendance_2026-11-16.csv
│       └── curriculum_attendance_review.csv
│
├── sample_classroom/               # Test classroom panorama images for offline demos
│
└── reports/                        # Project submission deliverables
    ├── MID_TERM_EVALUATION_REPORT.md # Official academic submission report
    ├── evaluation_results.json     # Quantitative benchmark output (JSON)
    └── evaluation_summary.txt      # Formatted metrics & confusion matrix summary
```

---

## 📅 Semester 5 Curriculum & Timetable (Mon–Sat, 10 AM – 4 PM)

| Time Slot | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10:00 - 11:00 AM** | **CS301** (CV) | **CS304** (CN) | **CS303** (DBMS) | **CS302** (OS) | **CS301** (CV) | **CS305** (Seminar) |
| **11:15 - 12:15 PM** | **CS302** (OS) | **CS301** (CV) | **CS304** (CN) | **CS304** (CN) | **CS303** (DBMS) | **CS301** (Tutorial) |
| **12:15 - 01:00 PM** | *Lunch Break* | *Lunch Break* | *Lunch Break* | *Lunch Break* | *Lunch Break* | *Lunch Break* |
| **01:00 - 02:00 PM** | **CS303** (DBMS) | **CS302** (OS) | **CS301** (CV) | **CS303** (DBMS) | **CS304** (CN) | **CS304** (Lab) |
| **02:15 - 04:00 PM** | **CS305** (Lab) | **CS303** (Lab) | **CS305** (Project) | **CS301** (CV Lab) | **CS302** (OS Lab) | *Mentorship* |

- **Operating Hours:** Monday to Saturday from **10:00 AM to 04:00 PM**.
- **Auto-Resolution:** In `AUTO` mode, the camera scanner detects the current day and time to tag attendance directly to the active lecture!

---

## 🎮 How to Run the Project

### Option A: Launch Desktop GUI Dashboard (Recommended for Faculty Viva)
```powershell
python run.py --mode gui
```
- Interactive tabs:
  1. **📋 Today's Log:** Shows live records with Student ID, Name, Subject, Time, and Confidence.
  2. **📊 Subject Review (<75%):** Full curriculum matrix with percentage per subject and Defaulter alert tags.
  3. **📅 Timetable (10-4):** Interactive timetable viewer for Monday to Saturday.
  4. **👥 Enrolled (10+):** Student directory.
  5. **➕ Enroll:** Live webcam registration with 5 captures.
  6. **📈 Accuracy:** Biometric evaluation benchmark.

### Option B: Real-Time Live Webcam Scanner
```powershell
python run.py --mode live
```
- Automatically resolves the active subject from the timetable schedule.
- Override subject manually: `python run.py --mode live --course CS301`
- Shortcuts: **`Q`** Quit | **`S`** Snapshot | **`E`** Export CSV

### Option C: View Conducted Classes & Attendee Rosters (Who & How Many)
```powershell
python run.py --mode classes
```
- Displays each conducted lecture session, headcount (Total Present vs Enrolled), present attendee list (with roll numbers, timestamps, and confidence), and absentee list.
- Exports `all_classes_headcount_summary.csv`.

### Option D: View Curriculum Subject Review in Terminal
```powershell
python run.py --mode report
```
- Displays formatted subject-wise attendance percentages and `<75%` Shortage alerts in terminal and exports `curriculum_attendance_review.csv`.

### Option E: View Weekly Timetable & Active Slot Status
```powershell
python run.py --mode timetable
```

### Option F: Run Automated Unit & Timetable Test Suite
```powershell
python -m unittest tests/test_timetable_attendance.py
```

### Option G: Enroll a New Student
```powershell
python run.py --mode enroll
```

### Option H: Run Biometric Accuracy & Performance Benchmark
```powershell
python run.py --mode evaluate
```

---

## 📊 System Performance & Benchmark Summary

| Metric | Measured Result | Benchmark Standard |
| :--- | :---: | :---: |
| **Accuracy** | **97.00%** | > 90% |
| **Precision** | **97.96%** | > 90% |
| **Recall (Sensitivity)** | **96.00%** | > 90% |
| **F1-Score** | **96.97%** | > 90% |
| **False Acceptance Rate (FAR)** | **2.00%** | < 5% |
| **False Rejection Rate (FRR)** | **4.00%** | < 5% |
| **Inference Latency** | **34.2 ms** | < 50 ms |
| **Processing Throughput** | **29.2 FPS** | Real-time (> 24 FPS) |

---

## 📑 Mid-Term Submission Report
The updated academic evaluation report with schedule architecture, subject-wise database models, and review metrics is located at:  
👉 **[`reports/MID_TERM_EVALUATION_REPORT.md`](file:///c:/Users/Ashuu/Documents/Sem%205/Projects/CV%20Assignment/reports/MID_TERM_EVALUATION_REPORT.md)**
