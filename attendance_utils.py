import csv
from datetime import datetime
import os

ATTENDANCE_FILE = "attendance/Attendance.csv"


def mark_attendance(name):
    os.makedirs("attendance", exist_ok=True)

    with open(ATTENDANCE_FILE, "a+", newline="") as f:
        f.seek(0)
        existing_data = f.readlines()
        name_list = [line.split(",")[0] for line in existing_data]

        if name not in name_list:
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            writer = csv.writer(f)
            writer.writerow([name, date, time])
