import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import re

# CONFIG
NPY_DIR = "D:/deepfake-detection-challenge/npy_final"
OUTPUT_MODEL_DIR = "D:/deepfake-detection-challenge/models"
BATCHES_PER_SESSION = 4
IMAGE_SHAPE = (224, 224, 3)
SESSIONS = 20
START_SESSION = 12  # Start from session 12
EPOCHS_PER_SESSION = 3  # 👈 Set to 3 epochs

def load_npy_file(filepath):
    try:
        arr = np.load(filepath)
        if arr.ndim == 4 or arr.ndim == 3:
            if arr.shape[1:] != IMAGE_SHAPE:
                arr_resized = tf.image.resize(arr, IMAGE_SHAPE[:2]).numpy()
            else:
                arr_resized = arr
            return arr_resized
        else:
            print(f"⚠️ Skipping label or invalid file: {filepath} (shape: {arr.shape})")
            return None
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def get_sorted_npy_files():
    files = os.listdir(NPY_DIR)
    files = [f for f in files if f.endswith(".npy")]
    
    def extract_index(f):
        match = re.search(r'(\d+)', f)
        return int(match.group(1)) if match else -1

    files.sort(key=lambda x: extract_index(x))
    return files

def train_on_batches(batches, session_idx):
    print(f"✅ Training model for session {session_idx} on {len(batches)} batches...")
    all_images = np.concatenate(batches, axis=0)
    labels = np.random.randint(0, 2, size=(all_images.shape[0],))  # Dummy labels

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=IMAGE_SHAPE),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(all_images, labels, epochs=EPOCHS_PER_SESSION, batch_size=32, verbose=1)

    model_path = os.path.join(OUTPUT_MODEL_DIR, f"model_session_{session_idx}.h5")
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")

def main():
    npy_files = get_sorted_npy_files()
    total_files = len(npy_files)
    i = (START_SESSION - 1) * BATCHES_PER_SESSION
    session_count = START_SESSION

    while session_count <= SESSIONS and i < total_files:
        session_batches = []
        print(f"\n🚀 Training session {session_count}/{SESSIONS}")

        for _ in tqdm(range(BATCHES_PER_SESSION), desc=f"📦 Loading files for session {session_count}"):
            if i >= total_files:
                break
            filepath = os.path.join(NPY_DIR, npy_files[i])
            data = load_npy_file(filepath)
            if data is not None:
                session_batches.append(data)
            i += 1

        if session_batches:
            train_on_batches(session_batches, session_count)
        else:
            print(f"⚠️ Skipping session {session_count} ❌ No valid data found.")
        
        session_count += 1

if __name__ == "__main__":
    main()
