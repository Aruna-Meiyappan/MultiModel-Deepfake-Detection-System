import os
import whisper
import glob
import pandas as pd

def process_audios(audio_dir):
    model = whisper.load_model("base")
    results = []

    for dataset_type in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(audio_dir, dataset_type, label)
            if not os.path.exists(path):
                print(f"Path does not exist: {path}")  # Debugging statement
                continue
            
            for audio_file in glob.glob(os.path.join(path, '*.*')):
                print(f"Processing file: {audio_file}")  # Debugging statement
                result = model.transcribe(audio_file)
                results.append({
                    "file": os.path.basename(audio_file),
                    "true_labels": 1 if label == "real" else 0,
                    "pred_labels": 1  # Mock prediction, replace with actual
                })
    
    print(f"Total results generated: {len(results)}")  # Debugging statement
    return results

def save_results(results):
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/audio_results.csv", index=False)
    print("Audio results saved!")

def main():
    audio_dir = '/Users/aruna/Desktop/DF/df/audio'  # Updated directory
    results = process_audios(audio_dir)
    save_results(results)

if __name__ == '__main__':
    main() 