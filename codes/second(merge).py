import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths to your processed frames
REAL_DIR = "D:/deepfake-detection-challenge/frames/real"
FAKE_DIR = "D:/deepfake-detection-challenge/frames/fake"
SAVE_DIR = "D:/deepfake-detection-challenge/npy_final"

IMG_SIZE = 224
BATCH_SIZE = 3000

os.makedirs(SAVE_DIR, exist_ok=True)

def load_and_preprocess_images(directory, label):
    data = []
    count = 0
    batch_index = 0

    for filename in tqdm(os.listdir(directory), desc=f"Processing {label} frames"):
        filepath = os.path.join(directory, filename)
        img = cv2.imread(filepath)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            data.append((img, label))
            count += 1

            # Save batch
            if count % BATCH_SIZE == 0:
                save_batch(data, label, batch_index)
                data = []
                batch_index += 1

    # Save remaining
    if data:
        save_batch(data, label, batch_index)

def save_batch(data, label, index):
    X = np.array([x[0] for x in data])
    y = np.array([x[1] for x in data])
    np.save(os.path.join(SAVE_DIR, f"X_{'real' if label == 0 else 'fake'}_{index}.npy"), X)
    np.save(os.path.join(SAVE_DIR, f"y_{'real' if label == 0 else 'fake'}_{index}.npy"), y)
    print(f"✅ Saved batch {index} for {'REAL' if label == 0 else 'FAKE'}: {len(X)} samples")

# Load all real and fake images
load_and_preprocess_images(REAL_DIR, 0)  # Label 0 = REAL
load_and_preprocess_images(FAKE_DIR, 1)  # Label 1 = FAKE

print("✅ All frame batches saved successfully.")
