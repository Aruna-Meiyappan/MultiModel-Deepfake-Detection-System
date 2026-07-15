# Multimodal Deepfake Detection System

Detection pipeline combining CNN, LSTM, WaveNet, BERT, and YOLOv7 to flag manipulated audio-visual content.

## Results
- 85% classification accuracy on 10,000+ samples
- 20% reduction in misclassification errors through model ensembling and data preprocessing

## Tech Stack
Python · TensorFlow · PyTorch · YOLOv7 · BERT

## How to Run
\`\`\`bash
git clone https://github.com/Aruna-Meiyappan/MultiModel-Deepfake-Detection-System
cd MultiModel-Deepfake-Detection-System
pip install -r requirements.txt
python main.py
\`\`\`

## Note
A Streamlit demo was attempted but currently has image-loading issues due to dataset size limits on free hosting. Working on a lighter-weight demo version.
