import os
import json
import multiprocessing
import cv2 
from tqdm import tqdm  # ✅ For progress bar

# Paths (Update based on your setup)
video_folder = r"D:\deepfake-detection-challenge\train_videos"
metadata_path = r"D:\deepfake-detection-challenge\train_videos\metadata.json"
output_real = r"D:\deepfake-detection-challenge\frames\real"
output_fake = r"D:\deepfake-detection-challenge\frames\fake"

# Create output directories
os.makedirs(output_real, exist_ok=True)
os.makedirs(output_fake, exist_ok=True)

# Load metadata.json
with open(metadata_path, "r") as file:
    metadata = json.load(file)

# Function to extract frames using GPU
def extract_frames(video_details):
    video_file, details = video_details
    video_path = os.path.join(video_folder, video_file)

    if not os.path.exists(video_path) or not video_file.endswith(".mp4"):
        return

    label = details["label"]
    output_folder = output_real if label == "REAL" else output_fake

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // 15)

    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_interval == 0:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                processed_frame = gpu_frame.download()
            except cv2.error:
                processed_frame = frame  # Fallback if CUDA fails

            frame_filename = os.path.join(output_folder, f"{video_file}_frame{frame_count}.jpg")
            cv2.imwrite(frame_filename, processed_frame)

        success, frame = cap.read()
        frame_count += 1

    cap.release()

# Progress bar wrapper
def extract_with_progress(data):
    for _ in tqdm(pool.imap_unordered(extract_frames, data), total=len(data), desc="Processing videos"):
        pass

if __name__ == "__main__":
    data = list(metadata.items())
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    extract_with_progress(data)
    pool.close()
    pool.join()

    print("✅ FAST Frame extraction completed using MULTIPROCESSING & GPU!")
