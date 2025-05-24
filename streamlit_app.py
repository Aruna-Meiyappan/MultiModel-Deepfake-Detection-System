import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
import streamlit as st

# Function to load CSV results
def load_results(modal):
    if modal == "audio":
        filename = "/Users/aruna/Desktop/DF/df/audio/results/audio_results.csv"
    elif modal == "text":
        filename = "/Users/aruna/Desktop/DF/df/text/results/text_results.csv"
    elif modal == "image":
        filename = "/Users/aruna/Desktop/DF/df/images/results/image_results.csv"
    elif modal == "video":
        filename = "/Users/aruna/Desktop/DF/df/videos/results/video_prediction.csv"
    
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return df
    else:
        st.warning(f"No results found for {modal}. Please run the respective model script.")
        return None

# Function to evaluate model performance
def evaluate_results(df):
    if df is not None and "true_labels" in df.columns and "pred_labels" in df.columns:
        accuracy = accuracy_score(df["true_labels"], df["pred_labels"])
        precision = precision_score(df["true_labels"], df["pred_labels"], zero_division=0)  # Handle division by zero
        return accuracy, precision
    else:
        st.warning("The DataFrame is missing required columns or is empty.")
        return None, None

# Main Streamlit App
def main():
    st.set_page_config(page_title="Multimodal Deepfake Detection", layout="wide")

    st.title('🔍 Multimodal Deepfake Detection Results')

    # List of modalities
    modalities = ["image", "text", "audio", "video"]
    
    # Layout
    col1, col2 = st.columns(2)

    for index, modal in enumerate(modalities):
        df = load_results(modal)
        accuracy, precision = evaluate_results(df)

        with (col1 if index % 2 == 0 else col2):
            st.header(f"📌 {modal.capitalize()} Results")
            
            if df is not None:
                st.write(df.head())  # Show sample results
                if accuracy is not None and precision is not None:
                    st.write(f"✅ **Accuracy:** {accuracy:.2f} | **Precision:** {precision:.2f}")
                    
                    # Visualization
                    fig, ax = plt.subplots()
                    ax.bar(["Accuracy", "Precision"], [accuracy, precision], color=["blue", "green"])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Score')
                    ax.set_title(f'{modal.capitalize()} Performance Metrics')
                    st.pyplot(fig)
                else:
                    st.warning("Could not calculate accuracy and precision.")

if __name__ == '__main__':
    main()