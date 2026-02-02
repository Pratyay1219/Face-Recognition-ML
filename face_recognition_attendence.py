import cv2
import face_recognition
import numpy as np
from face_encodings import load_encodings, classNames
from attendance_utils import mark_attendance

cap = cv2.VideoCapture(0)

print("Starting Face Attendance System...")
knownEncodings = load_encodings()

THRESHOLD = 0.45

while True:
    success, img = cap.read()
    if not success:
        print("Failed to access camera")
        break

    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(img_rgb)
    encodings = face_recognition.face_encodings(img_rgb, faces)

    for encode_face, face_loc in zip(encodings, faces):

        distances = face_recognition.face_distance(knownEncodings, encode_face)
        match_index = np.argmin(distances)

        name = "UNKNOWN"
        status = ""
        color = (0, 0, 255)

        if distances[match_index] < THRESHOLD:
            name = classNames[match_index].upper()
            color = (0, 255, 0)

            marked = mark_attendance(name)

            if marked:
                status = "MARKED"
            else:
                status = "ALREADY MARKED"

        confidence = int((1 - distances[match_index]) * 100)

        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{name} {status} {confidence}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Attendance", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
