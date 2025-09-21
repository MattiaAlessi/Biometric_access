import cv2
import os
import numpy as np

def train_model(dataset_path="dataset", model_path="recognizer.yml"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for user in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user)
        label_map[current_label] = user
        for img_file in os.listdir(user_path):
            img_path = os.path.join(user_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(current_label)
        current_label += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)
    print("✅ Modello LBPH addestrato e salvato.")
    return label_map
