import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# --- Config ---
DATA_DIR = 'D:/deepfake-detection-challenge/npy_final'
MODEL_SAVE_DIR = 'D:/deepfake-detection-challenge/models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS_PER_SESSION = 3
NUM_SESSIONS = 20
IMG_SHAPE = (224, 224, 3)

# --- Model Architecture ---
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Load and Validate Images ---
def load_images_and_labels(file_path):
    try:
        X = np.load(file_path, allow_pickle=True)

        # Debug: print actual shape
        print(f"🔍 Loaded {os.path.basename(file_path)} with shape: {X.shape}")

        # Fix shape if needed
        if X.ndim == 3:  # missing channel dimension
            X = np.expand_dims(X, axis=-1)

        if X.shape[1:] != IMG_SHAPE:
            print(f"⚠️ Resizing {os.path.basename(file_path)} to {IMG_SHAPE}")
            X = tf.image.resize(X, IMG_SHAPE[:2]).numpy()

        fname = os.path.basename(file_path).lower()
        label = 1 if 'fake' in fname else 0
        y = np.full((X.shape[0],), label, dtype=np.uint8)
        return X, y

    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None, None

# --- Gather .npy Files ---
all_files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith('.npy') and ('real' in f.lower() or 'fake' in f.lower())
])

files_per_session = len(all_files) // NUM_SESSIONS

# --- Detect last completed session ---
existing_models = [
    f for f in os.listdir(MODEL_SAVE_DIR)
    if f.startswith('model_session_') and f.endswith('.h5')
]
completed_sessions = sorted([
    int(f.split('_')[-1].split('.')[0]) for f in existing_models
])
start_session = max(completed_sessions, default=0)

# --- Resume Training ---
model = build_model()
if start_session > 0:
    latest_model_path = os.path.join(MODEL_SAVE_DIR, f'model_session_{start_session}.h5')
    if os.path.exists(latest_model_path):
        model.load_weights(latest_model_path)
        print(f"📥 Loaded weights from: {latest_model_path}")

# --- Train Remaining Sessions ---
for session in range(start_session, NUM_SESSIONS):
    print(f"\n🚀 Training session {session + 1}/{NUM_SESSIONS}")

    start_idx = session * files_per_session
    end_idx = (session + 1) * files_per_session
    session_files = all_files[start_idx:end_idx]

    X_all, y_all = [], []
    for file in tqdm(session_files, desc=f"📦 Loading files for session {session + 1}"):
        file_path = os.path.join(DATA_DIR, file)
        X, y = load_images_and_labels(file_path)
        if X is not None and y is not None:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        print(f"⚠️ Skipping session {session + 1} ❌ No valid data found.")
        continue

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    X_all, y_all = shuffle(X_all, y_all, random_state=42)

    model.fit(X_all, y_all, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_SESSION, verbose=1)

    save_path = os.path.join(MODEL_SAVE_DIR, f'model_session_{session + 1}.h5')
    model.save(save_path)
    print(f"✅ Saved model: {save_path}")
