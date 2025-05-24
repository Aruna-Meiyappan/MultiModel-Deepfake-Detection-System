import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov7'))

import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.datasets import letterbox
from models.yolo import Model
from torch.nn import Sequential
from torch.serialization import load as torch_load, add_safe_globals

import cv2
import glob
import pandas as pd
import numpy as np

# Allow YOLOv7 model and Sequential for safe loading
add_safe_globals([Model, Sequential])

# Load the model
model_path = '/Users/aruna/Desktop/DF/df/videos/yolov7/yolov7.pt'  # Swap back to yolov7s.pt if confident
device = torch.device('cpu')

# Manual model loading for safety
ckpt = torch_load(model_path, map_location=device, weights_only=False)
model = ckpt['model'].float().to(device)
model.eval()

def preprocess_video_frame(frame):
    # Preprocess a single video frame to the YOLOv7 input format
    frame = letterbox(frame, new_shape=640)[0]
    frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    frame = np.ascontiguousarray(frame)
    frame = torch.from_numpy(frame).to(device).float()
    frame /= 255.0
    if frame.ndimension() == 3:
        frame = frame.unsqueeze(0)
    return frame

def process_videos(data_dir, max_videos=5):  # Limit added
    results = []
    count = 0

    for label in ['real', 'fake']:
        path = os.path.join(data_dir, label)
        if not os.path.exists(path):
            continue

        for video_file in glob.glob(os.path.join(path, '*.mp4')):  # Only looking for .mp4 videos
            if count >= max_videos:
                return results  # Return early if the limit is reached

            video_capture = cv2.VideoCapture(video_file)
            if not video_capture.isOpened():
                continue

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                input_tensor = preprocess_video_frame(frame)
                with torch.no_grad():  # Important memory optimization
                    pred = model(input_tensor, augment=False)[0]
                    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

                pred_label = 1 if len(pred[0]) > 0 else 0  # Assuming detection means 'real'
                results.append({
                    "video": os.path.basename(video_file),
                    "true_labels": 1 if label == "real" else 0,
                    "pred_labels": pred_label
                })
                count += 1

            video_capture.release()

    return results

def save_results(results):
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/video_prediction.csv", index=False)

    with open("video_predictions.txt", "w") as f:
        for row in results:
            f.write(f"{row['video']}: {row['pred_labels']}\n")

    print("✅ Video results saved to results/video_prediction.csv and video_predictions.txt")

def main():
    videos_dir = '/Users/aruna/Desktop/DF/df/videos'  # Path to your video dataset
    results = process_videos(videos_dir)
    save_results(results)

if __name__ == '__main__':
    main()
