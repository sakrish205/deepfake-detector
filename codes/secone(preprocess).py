import os
import cv2
import numpy as np
from tqdm import tqdm

# Set your paths
REAL_DIR = "D:/deepfake-detection-challenge/frames/real"
FAKE_DIR = "D:/deepfake-detection-challenge/frames/fake"
NPY_DIR = "D:/deepfake-detection-challenge/npy"
IMG_SIZE = 224
BATCH_SIZE = 500  # Save every 500 frames

# Create output directory if not exists
os.makedirs(NPY_DIR, exist_ok=True)

# Function to load and save images in batches
def load_and_save_in_batches(directory, label, label_name):
    X_batch, y_batch = [], []
    file_list = os.listdir(directory)
    failed = 0
    total_saved = 0
    batch_num = 0

    for i, filename in enumerate(tqdm(file_list, desc=f"Processing {label_name}")):
        filepath = os.path.join(directory, filename)
        img = cv2.imread(filepath)

        if img is None:
            failed += 1
            continue

        try:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            X_batch.append(img)
            y_batch.append(label)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            failed += 1
            continue

        # Save in batches
        if len(X_batch) >= BATCH_SIZE:
            np.save(os.path.join(NPY_DIR, f"X_{label_name}_{batch_num}.npy"), np.array(X_batch))
            np.save(os.path.join(NPY_DIR, f"y_{label_name}_{batch_num}.npy"), np.array(y_batch))
            print(f"✅ Saved batch {batch_num} with {len(X_batch)} images")
            total_saved += len(X_batch)
            X_batch, y_batch = [], []
            batch_num += 1

    # Save remaining
    if X_batch:
        np.save(os.path.join(NPY_DIR, f"X_{label_name}_{batch_num}.npy"), np.array(X_batch))
        np.save(os.path.join(NPY_DIR, f"y_{label_name}_{batch_num}.npy"), np.array(y_batch))
        print(f"✅ Saved final batch {batch_num} with {len(X_batch)} images")
        total_saved += len(X_batch)

    print(f"🎉 Finished {label_name}. Total saved: {total_saved}, Failed: {failed}")
    return total_saved

# Process both real and fake folders
total_real = load_and_save_in_batches(REAL_DIR, 0, "real")
total_fake = load_and_save_in_batches(FAKE_DIR, 1, "fake")

print(f"\n🟢 All preprocessing done!\nReal images: {total_real}, Fake images: {total_fake}")
print(f"🗂️ Saved .npy batches to: {NPY_DIR}")
