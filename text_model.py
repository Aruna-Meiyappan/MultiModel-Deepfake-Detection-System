import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score

def load_data():
    real_data = pd.read_csv('real/real_texts.csv', header=None, names=['text'])
    fake_data = pd.read_csv('fake/fake_texts.csv', header=None, names=['text'])

    real_data['label'] = 1  # Real label
    fake_data['label'] = 0  # Fake label

    return pd.concat([real_data, fake_data], ignore_index=True)

def train_model(data):
    results = []
    for index, row in data.iterrows():
        results.append({
            "file": f"text_{index}.txt",
            "true_labels": row["label"],
            "pred_labels": 1  # Mock prediction, replace with actual
        })
    
    return results

def save_results(results):
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/text_results.csv", index=False)
    print("Text results saved!")

def main():
    
    data = load_data()
    results = train_model(data)
    save_results(results)

if __name__ == '__main__':
    main()


