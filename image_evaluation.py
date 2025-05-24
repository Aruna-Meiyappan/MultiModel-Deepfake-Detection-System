# image_evaluation.py
import os
from sklearn.metrics import accuracy_score, precision_score

def evaluate_model(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    return accuracy, precision

def load_true_labels(image_dir):
    """Dynamically assigns labels based on folder names (real = 1, fake = 0)."""
    true_labels = {}
    for dataset_type in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(image_dir, dataset_type, label)
            if not os.path.exists(path):
                continue

            for image_file in os.listdir(path):
                true_labels[image_file] = 1 if label == "real" else 0

    return true_labels

def load_predictions():
    """Load predictions from the image_predictions.txt file."""
    predictions = {}
    try:
        with open("image_predictions.txt", "r") as f:
            for line in f:
                filename, label = line.strip().split(": ")
                predictions[filename] = int(label)
    except FileNotFoundError:
        print("Error: image_predictions.txt not found. Run image_model.py first.")
        return None
    return predictions

def main():
    image_dir = "images"
    true_labels = load_true_labels(image_dir)
    predictions = load_predictions()

    if predictions is None:
        return

    y_true = [true_labels.get(img, 0) for img in predictions.keys()]
    y_pred = list(predictions.values())

    accuracy, precision = evaluate_model(y_pred, y_true)
    print(f"Image Evaluation Results - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}")

if __name__ == '__main__':
    main()
