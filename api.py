from fastapi import FastAPI
import cv2
import face_recognition
import numpy as np
from face_encodings import knownEncodings, classNames
from attendance_utils import mark_attendance

app = FastAPI()

THRESHOLD = 0.45

@app.get("/")
def home():
    return {"message": "AI Attendance System Running"}

@app.post("/mark")
def mark():
    cap = cv2.VideoCapture(0)

    success, img = cap.read()
    cap.release()

    if not success:
        return {"error": "Camera not available"}

    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(img_rgb)
    encodings = face_recognition.face_encodings(img_rgb, faces)

    if len(encodings) == 0:
        return {"result": "No face detected"}

    encode_face = encodings[0]
    distances = face_recognition.face_distance(knownEncodings, encode_face)
    match_index = np.argmin(distances)

    if distances[match_index] < THRESHOLD:
        name = classNames[match_index].upper()
        mark_attendance(name)
        return {"result": f"{name} marked successfully"}
    else:
        return {"result": "Unknown face"}
