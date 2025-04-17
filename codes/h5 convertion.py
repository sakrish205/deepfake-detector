import tensorflow as tf
import numpy as np
import os

model_dir = r"D:\deepfake-detection-challenge\models"
model_paths = [os.path.join(model_dir, f"model_session_{i}.h5") for i in range(1, 12)]

print("📦 Loading base model...")
merged_model = tf.keras.models.load_model(model_paths[0])
model_weights = [tf.keras.models.load_model(path).get_weights() for path in model_paths]

print("🧠 Averaging weights...")
avg_weights = []
skipped = 0

for i, weights_per_layer in enumerate(zip(*model_weights)):
    shapes = [w.shape for w in weights_per_layer]
    if all(s == shapes[0] for s in shapes):
        avg = np.mean(weights_per_layer, axis=0)
        avg_weights.append(avg)
    else:
        print(f"⚠️ Skipping layer {i} due to shape mismatch: {shapes}")
        avg_weights.append(weights_per_layer[0])  # fallback to model 1's weights
        skipped += 1

print(f"✅ Averaging done. Skipped {skipped} mismatched layers.")

merged_model.set_weights(avg_weights)

# Save merged model
merged_model_path = os.path.join(model_dir, "merged_model.h5")
merged_model.save(merged_model_path)
print(f"✅ Merged model saved at: {merged_model_path}")

# Convert to TFLite
print("⚙️ Converting to TFLite format (CPU-only)...")
converter = tf.lite.TFLiteConverter.from_keras_model(merged_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# Save the TFLite model
tflite_path = os.path.join(model_dir, "merged_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"🎉 TFLite model saved at: {tflite_path}")
