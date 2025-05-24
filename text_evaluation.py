import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score

def load_results():
    """Load results from the saved CSV file."""
    results_path = "results/text_results.csv"
    if not os.path.exists(results_path):
        print("Error: Results file not found! Run text_model.py first.")
        return None
    return pd.read_csv(results_path)

def evaluate_model(results):
    """Evaluate the model based on the results."""
    y_true = results['true_labels']
    y_pred = results['pred_labels']

    # Calculate accuracy and precision
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    print(f"Evaluation Results - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}")

def main():
    results = load_results()
    if results is not None:
        evaluate_model(results)

if __name__ == '__main__':
    main()