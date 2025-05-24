# image_model.py
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
model_path = '/Users/aruna/Desktop/DF/df/images/yolov7/yolov7.pt'  # Swap back to yolov7s.pt if confident
device = torch.device('cpu')

# Manual model loading for safety
ckpt = torch_load(model_path, map_location=device, weights_only=False)
model = ckpt['model'].float().to(device)
model.eval()

def preprocess_image(img):
    img = letterbox(img, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def process_images(data_dir, max_images=5):  # Limit added
    results = []
    count = 0

    for dataset_type in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(data_dir, dataset_type, label)
            if not os.path.exists(path):
                continue

            for image_file in glob.glob(os.path.join(path, '*.*')):
                if count >= max_images:
                    return results  # Return early if limit reached

                img = cv2.imread(image_file)
                if img is None:
                    continue

                input_tensor = preprocess_image(img)
                with torch.no_grad():  # Important memory optimization
                    pred = model(input_tensor, augment=False)[0]
                    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

                pred_label = 1 if len(pred[0]) > 0 else 0
                results.append({
                    "file": os.path.basename(image_file),
                    "true_labels": 1 if label == "real" else 0,
                    "pred_labels": pred_label
                })
                count += 1

    return results

def save_results(results):
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/image_results.csv", index=False)

    with open("image_predictions.txt", "w") as f:
        for row in results:
            f.write(f"{row['file']}: {row['pred_labels']}\n")

    print("✅ Image results saved to results/image_results.csv and image_predictions.txt")

def main():
    images_dir = '/Users/aruna/Desktop/DF/df/images'
    results = process_images(images_dir)
    save_results(results)

if __name__ == '__main__':
    main()
