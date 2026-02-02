import cv2
import face_recognition
import os

DATASET_PATH = "Dataset"

images = []
classNames = []

# Load images only once
for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        images.append(img)
        classNames.append(person)


def find_encodings(images):
    encode_list = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encode_list.append(encodes[0])

    return encode_list


# Lazy-load encodings (important for Streamlit)
_knownEncodings = None

def load_encodings():
    global _knownEncodings
    if _knownEncodings is None:
        _knownEncodings = find_encodings(images)
        print(f"Encodings Loaded Successfully for {len(_knownEncodings)} faces")
    return _knownEncodings
