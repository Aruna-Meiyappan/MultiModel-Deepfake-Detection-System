import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score

def evaluate_model(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    return accuracy, precision

def load_true_labels(audio_dir):
    """Dynamically assigns labels based on folder names (real = 1, fake = 0)."""
    true_labels = {}
    for dataset_type in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(audio_dir, dataset_type, label)
            if not os.path.exists(path):
                print(f"Path does not exist: {path}")  # Debugging statement
                continue

            for audio_file in os.listdir(path):
                true_labels[audio_file] = 1 if label == "real" else 0

    return true_labels

def load_predictions():
    """Load predictions from the audio_results.csv file."""
    predictions = {}
    try:
        df = pd.read_csv("results/audio_results.csv")  # Update the path to the CSV file
        for index, row in df.iterrows():
            predictions[row['file']] = row['pred_labels']  # Use the correct column names
    except FileNotFoundError:
        print("Error: audio_results.csv not found. Run audio_model.py first.")
        return None
    return predictions

def main():
    audio_dir = '/Users/aruna/Desktop/DF/df/audio'  # Updated directory
    true_labels = load_true_labels(audio_dir)
    predictions = load_predictions()

    if predictions is None:
        return

    y_true = [true_labels.get(audio, 0) for audio in predictions.keys()]
    y_pred = list(predictions.values())

    accuracy, precision = evaluate_model(y_pred, y_true)
    print(f"Audio Evaluation Results - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}")

if __name__ == '__main__':
    main()