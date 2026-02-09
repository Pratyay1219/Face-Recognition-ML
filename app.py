import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

from face_encodings import load_encodings, classNames
from attendance_utils import mark_attendance

# Page setup
st.set_page_config(page_title="AI Face Attendance", layout="centered")
st.title("🎓 AI Face Attendance System")
st.write("Look at the camera and take a photo to mark attendance.")

# Load encodings once
knownEncodings = load_encodings()

# Webcam input
photo = st.camera_input("Take a photo")

if photo is not None:
    image = Image.open(photo)
    image_np = np.array(image)

    # Detect faces
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    if len(face_encodings) == 0:
        st.error("❌ No face detected. Try again.")
    else:
        encode = face_encodings[0]

        distances = face_recognition.face_distance(knownEncodings, encode)
        match_index = np.argmin(distances)

        if distances[match_index] < 0.45:
            name = classNames[match_index].upper()
            mark_attendance(name)
            st.success(f"✅ {name} marked successfully")
        else:
            st.error("❌ Unknown face detected")
