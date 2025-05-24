import os
from sklearn.metrics import accuracy_score, precision_score

def evaluate_model(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    return accuracy, precision

def load_true_labels(video_dir):
    labels = {}
    for dataset_type in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(video_dir, dataset_type, label)
            if not os.path.exists(path): continue
            for video_file in os.listdir(path):
                labels[video_file] = 1 if label == "real" else 0
    return labels

def load_predictions():
    predictions = {}
    try:
        with open("video_predictions.txt", "r") as f:
            for line in f:
                filename, label = line.strip().split(": ")
                predictions[filename] = int(label)
    except FileNotFoundError:
        print("Error: video_predictions.txt not found. Run video_model.py first.")
        return None
    return predictions

def main():
    video_dir = "/Users/aruna/Desktop/DF/df/videos"
    true_labels = load_true_labels(video_dir)
    predictions = load_predictions()
    if predictions is None: return

    y_true = [true_labels.get(vid, 0) for vid in predictions.keys()]
    y_pred = list(predictions.values())

    accuracy, precision = evaluate_model(y_pred, y_true)
    print(f"🎯 Video Evaluation - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}")

if __name__ == '__main__':
    main()
