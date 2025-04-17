import tensorflow as tf
import numpy as np
import cv2
import os

# Load TFLite model
model_path = r"D:\deepfake-detection-challenge\final models\merged_model.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print("Input shape expected:", input_shape)

# Load and preprocess the image
img_path = r"D:\deepfake-detection-challenge\frames\fake\aagfhgtpmv.mp4_frame197.jpg"  # <-- Your image path
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

img = cv2.imread(img_path)
img = cv2.resize(img, (input_shape[1], input_shape[2]))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# Output
output = interpreter.get_tensor(output_details[0]['index'])
print("Raw model output:", output)

# Prediction based on sigmoid output
pred = 1 if output[0][0] > 0.5 else 0
print("Prediction:", "Fake" if pred == 1 else "Real")
print("Confidence score:", float(output[0][0]))
