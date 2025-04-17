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
def predict_image(img_path, threshold=0.5):
    img = cv2.imread(img_path)
    if img is None:
        return "Unreadable", 0.0, None

    img_resized = cv2.resize(img, (img_width, img_height))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(output[0][0])
    label = "Fake" if confidence > threshold else "Real"
    return label, confidence, img_resized

# === RUN ON USER-INPUT IMAGE ===
if os.path.exists(single_image_path):
    label, confidence, preview_img = predict_image(single_image_path)

    print(f"\nResult:")
    print(f"File:        {os.path.basename(single_image_path)}")
    print(f"Prediction:  {label}")
    print(f"Confidence:  {confidence:.4f}")
    print(f"Threshold:   0.50\n")

    # === Optional: Show image with overlay ===
    if preview_img is not None:
        overlay = preview_img.copy()
        text = f"{label} ({confidence:.2%})"
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Prediction", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print(f"Image not found: {single_image_path}")
