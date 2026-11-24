"""
Data Reset Tool.
Clears all stored students, facial vectors, and attendance records.
Ensures ONLY real enrolled students are stored on the platform.
"""
from src.database import reset_all_data

if __name__ == "__main__":
    reset_all_data()
