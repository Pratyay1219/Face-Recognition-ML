import cv2
import face_recognition
import os

DATASET_PATH = "dataset"

images = []
classNames = []

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        images.append(img)
        classNames.append(person)


def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


knownEncodings = find_encodings(images)

print("Encodings Loaded Successfully")
