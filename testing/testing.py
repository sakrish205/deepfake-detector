import tensorflow as tf
import numpy as np
import cv2
import os

# === CONFIG ===
model_path = r"D:\deepfake-detection-challenge\final models\merged_model.tflite"
single_image_path = input("Enter the full path to the image file: ").strip('"')

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

# === RUN ON USER-INPUT IMAGE ===
if os.path.exists(single_image_path):
    label, confidence = predict_image(single_image_path)
    print(f"{os.path.basename(single_image_path):40s} ➜ {label:5s} (Confidence: {confidence:.4f})")
else:
    print(f"Image not found: {single_image_path}")
