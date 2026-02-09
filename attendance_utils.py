import csv
from datetime import datetime
import os

# Absolute path of this file (Face-Recognition-ML folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to attendance folder
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "Attendance.csv")

# Stores last marked time of each person
last_marked = {}

def mark_attendance(name):
    global last_marked

    os.makedirs(ATTENDANCE_DIR, exist_ok=True)

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # ⏱️ If already marked within 60 seconds → block
    if name in last_marked:
        diff = (now - last_marked[name]).total_seconds()
        if diff < 60:
            return False   # Already marked in last 1 min

    # Update last marked time
    last_marked[name] = now

    # Create file if it does not exist
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # Append attendance
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, time])

    print(f"Attendance marked for {name} at {time}")
    print("Saved at:", ATTENDANCE_FILE)

    return True
