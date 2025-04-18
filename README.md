# Deepfake Detection Challenge

![Deepfake Detection](https://img.shields.io/badge/AI-Deepfake%20Detection-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A web application that can detect deepfake images using machine learning and TensorFlow Lite.

## 📝 Overview

This project provides an easy-to-use web interface for detecting deepfake images. The application uses a TensorFlow Lite model trained to identify visual artifacts and inconsistencies commonly found in deepfake media, providing a confidence score for each prediction.

## ✨ Features

- 🖥️ Web-based interface for easy access
- ⚡ Real-time deepfake detection
- 📊 Confidence score for predictions
- 🖼️ Support for common image formats (JPG, PNG, JPEG)

## 🛠️ Technical Details

The application is built using:
- Python Flask for the web framework
- TensorFlow and TensorFlow Lite for the machine learning model
- OpenCV for image processing
- HTML/CSS for the front-end interface

## 📋 Project Structure

```
├── app.py                  # Main Flask application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── codes/                  # Training and preprocessing scripts
├── final models/           # Trained ML models
│   ├── merged_model.h5
│   └── merged_model.tflite
├── static/                 # Static assets
│   ├── bg.jpg
│   └── uploads/            # Folder for uploaded images
└── templates/              # HTML templates
    ├── index.html
    ├── loading.html
    └── result.html
```

## 🚀 Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/deepfake-detection-challenge.git
   cd deepfake-detection-challenge
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## 🚀 Quick Start
For users familiar with Python and Flask:
```bash
git clone https://github.com/sakrish205/deepfake-detector.git && cd deepfake-detector
pip install -r requirements.txt
python app.py
```

## 💻 How to Use

1. Access the web interface at `http://localhost:5000`
2. Click "Choose File" to upload an image
3. Click "Upload" to analyze the image
4. View the results showing whether the image is classified as real or fake, along with a confidence score

## 🎬 Demo

[Video demonstration will be uploaded soon]

## 🔮 Future Improvements

- 📹 Support for video analysis
- 📦 Batch processing of multiple files
- 📊 Enhanced visualizations of detection results
- 📱 Mobile application development

## 👥 Contributors

- [Saketha Krishna](https://github.com/sakrish205)
- [Berlin Selvia](https://github.com/berlincodez)
- [Tamil Selvan](https://github.com/tamilselvan-s-d)
- [Suren](https://github.com/Suren-GPU)
- [Varun]()

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
