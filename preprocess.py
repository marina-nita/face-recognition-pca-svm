import cv2
import numpy as np
import os

def preprocess_image(path, size=(100, 100)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {path} could not be read.")
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, size)
    return img.flatten()

def load_dataset(directory, size=(100, 100)):
    data, labels = [], []
    for label in os.listdir(directory):
        folder = os.path.join(directory, label)
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            try:
                img = preprocess_image(path, size)
                data.append(img)
                labels.append(label)
            except:
                continue
    return np.array(data), np.array(labels)
