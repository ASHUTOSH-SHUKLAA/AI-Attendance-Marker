"""
Configuration settings for the AI-Powered Classroom Attendance System.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STUDENTS_DIR = DATA_DIR / "students"
ENCODINGS_FILE = DATA_DIR / "encodings.pkl"
DATABASE_FILE = DATA_DIR / "attendance.db"
ATTENDANCE_LOGS_DIR = DATA_DIR / "attendance_logs"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DIR = BASE_DIR / "sample_classroom"

# Ensure essential directories exist
for folder in [DATA_DIR, STUDENTS_DIR, ATTENDANCE_LOGS_DIR, REPORTS_DIR, MODELS_DIR, SAMPLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Face Recognition Settings
# Tolerance determines strictness: lower is stricter (less false positives, higher false negatives)
# Default for dlib 128D is 0.50 - 0.60
FACE_DISTANCE_TOLERANCE = 0.52

# Recognition Model: 'hog' (faster, good for CPU) or 'cnn' (more accurate, requires GPU/CUDA)
DETECTION_MODEL = "hog"

# Multi-sampling: number of times to re-sample face for jittering during encoding
NUM_JITTERS = 1

# Camera & Video Stream Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
PROCESS_EVERY_N_FRAMES = 2  # Skip frames for real-time high-FPS performance

# Attendance Logic Settings
# Cooldown/Debounce time in seconds before re-logging the same student in the same run
DEBOUNCE_SECONDS = 300  # 5 minutes

# Default lecture/session name
DEFAULT_COURSE_CODE = "CS301"
DEFAULT_SESSION_NAME = "Lecture_01"

# Academic Schedule & Timetable Settings
SCHEDULE_OPERATING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
SCHEDULE_START_TIME = "10:00"  # 10:00 AM
SCHEDULE_END_TIME = "16:00"    # 04:00 PM
ENFORCE_TIME_WINDOW = True     # Strict timetable enforcement by default
DEMO_MODE = False              # Set True or use --demo / GUI toggle to bypass schedule for demo testing
MIN_ATTENDANCE_PERCENTAGE = 75.0  # College eligibility threshold

# Visual Display Settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2
COLOR_KNOWN = (0, 200, 0)       # Green (BGR)
COLOR_UNKNOWN = (0, 0, 220)     # Red (BGR)
COLOR_TEXT = (255, 255, 255)    # White
COLOR_BANNER = (40, 40, 40)     # Dark Gray
COLOR_HIGHLIGHT = (255, 165, 0) # Orange / Cyan
