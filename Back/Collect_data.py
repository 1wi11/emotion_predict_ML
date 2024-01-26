import csv
import os
import cv2
import mediapipe as mp
import glob
import numpy as np

category = ['happy', 'anger', 'neutral', 'sad', 'surprise']
image_files = []
labels = []

# 각각의 감정 상황 이미지 경로 추가
img_happiness = glob.glob("./datasets/happy/*")
img_anger = glob.glob("./datasets/anger/*")
img_neutral = glob.glob("./datasets/neutral/*")
img_sad = glob.glob("./datasets/sad/*")
img_surprise = glob.glob("./datasets/surprise/*")

# 이미지 파일과 레이블 할당
for img_path in img_happiness:
    image_files.append(img_path)
    labels.append(category.index('happy'))

for img_path in img_anger:
    image_files.append(img_path)
    labels.append(category.index('anger'))

for img_path in img_neutral:
    image_files.append(img_path)
    labels.append(category.index('neutral'))

for img_path in img_sad:
    image_files.append(img_path)
    labels.append(category.index('sad'))

for img_path in img_surprise:
    image_files.append(img_path)
    labels.append(category.index('surprise'))


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def extract_landmarks(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            return [(landmark.x, landmark.y) for landmark in landmarks]

        return None

# 랜드마크 표시 및 정규화하여 CSV 파일에 저장
header = ['emotion'] + [f'landmark_{i}' for i in range(468)]
output_file = './model/keypoint_classifier/keypoint_classifier_label.csv'

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for image_file, label in zip(image_files, labels):
        image = cv2.imread(image_file)

        # 이미지 불러오기가 실패한 경우 스킵
        if image is None:
            continue

        landmarks = extract_landmarks(image)

        if landmarks:
            # 랜드마크 값을 정규화
            landmarks = np.array(landmarks)
            landmarks = (landmarks - landmarks.min(axis=0)) / (landmarks.max(axis=0) - landmarks.min(axis=0))
            
            # 레이블과 랜드마크 값을 함께 저장
            row = [label] + landmarks[:, :2].flatten().tolist()
            writer.writerow(row)