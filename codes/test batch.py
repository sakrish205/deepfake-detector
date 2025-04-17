import tensorflow as tf
import numpy as np
import cv2
import os

# === CONFIG ===
model_path = r"D:\deepfake-detection-challenge\final models\merged_model.tflite"
image_folder = r"D:\deepfake-detection-challenge\test\samples"  # change to your folder (real/fake)

# === LOAD MODEL ===
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
img_height, img_width = input_shape[1], input_shape[2]

# === PREDICT FUNCTION ===
def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Unreadable", 0.0
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(output[0][0])
    label = "Fake" if confidence > 0.5 else "Real"
    return label, confidence

# === RUN ON FOLDER ===
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Testing {len(image_files)} images in: {image_folder}\n")

for img_file in image_files:
    full_path = os.path.join(image_folder, img_file)
    label, confidence = predict_image(full_path)
    print(f"{img_file:50s} ➜ {label:5s} (Confidence: {confidence:.4f})")
