from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="final models/merged_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Validate file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image to match model input
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('loading', filename=filename))
    return redirect(url_for('index'))

@app.route('/loading/<filename>')
def loading(filename):
    return render_template('loading.html', filename=filename)

@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = preprocess_image(filepath)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Assume output[0][0] is the fake probability
    fake_prob = float(output_data[0][0])
    prediction = "Fake" if fake_prob >= 0.5 else "Real"
    confidence = round(fake_prob * 100 if prediction == "Fake" else (100 - fake_prob * 100), 2)

    return render_template('result.html', prediction=prediction, confidence=confidence, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
