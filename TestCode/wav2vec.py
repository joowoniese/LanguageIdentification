import os
import glob
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["HF_HOME"] = "/home/joowoniese/huggingface_cache"

model_name = "facebook/wav2vec2-large-xlsr-53"

# Feature Extractor and Model
print("Loading feature extractor and model...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).to(device)

data_dir = "/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/Train/"
sub_dirs = [os.path.join(data_dir, sub_dir) for sub_dir in os.listdir(data_dir)]

audio_files = []
labels = []

for sub_dir in sub_dirs:
    label = os.path.basename(sub_dir)  
    files = glob.glob(os.path.join(sub_dir, "*.wav"))
    audio_files.extend(files)
    labels.extend([label] * len(files))

print(f"Found {len(audio_files)} audio files across {len(sub_dirs)} labels: {set(labels)}")

features_list = []
file_names_list = []  
labels_list = []

for i, file_path in enumerate(audio_files, start=1):
    try:
        file_name = os.path.basename(file_path)
        label = labels[i - 1]

        print(f"[{i}/{len(audio_files)}] Processing file: {file_name} (Label: {label})")

        waveform, sample_rate = torchaudio.load(file_path)

        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        waveform = waveform.mean(dim=0)  # (samples,)
        waveform = waveform.unsqueeze(0)  # (1, samples)

        inputs = feature_extractor(waveform.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)

        inputs = {key: val.to(device) for key, val in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state

        features = hidden_states.mean(dim=1).squeeze()
        features_list.append(features.cpu().numpy())
        file_names_list.append(file_name)  # 파일 이름 추가
        labels_list.append(label)

        print(f"[{i}/{len(audio_files)}] Successfully processed file: {file_name}")

    except Exception as e:
        print(f"[{i}/{len(audio_files)}] Error processing {file_name}: {e}")

print("Saving features, labels, and file names...")
output_dir = "/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/Train/wav2vec_featuredata/"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "audio_features.npy"), np.array(features_list))
np.save(os.path.join(output_dir, "file_names.npy"), np.array(file_names_list))  # 파일 이름 저장
pd.DataFrame({"File": file_names_list, "Label": labels_list}).to_csv(os.path.join(output_dir, "audio_labels.csv"), index=False)

print(f"Features saved to {os.path.join(output_dir, 'audio_features.npy')}")
print(f"File names saved to {os.path.join(output_dir, 'file_names.npy')}")
print(f"Labels saved to {os.path.join(output_dir, 'audio_labels.csv')}")

features = np.load(os.path.join(output_dir, "audio_features.npy"))
file_names = np.load(os.path.join(output_dir, "file_names.npy"))
labels = pd.read_csv(os.path.join(output_dir, "audio_labels.csv"))["Label"].values


label_names = {
    "chinese": "Chinese",
    "japanese": "Japanese",
    "french": "French",
    "spanish": "Spanish",
    "korean": "Korean"
}

def tsne_visualization(features, labels, output_dir, label_names):
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels))) 

    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)
        plt.scatter(
            tsne_features[idx, 0], tsne_features[idx, 1],
            label=label_names[label], 
            alpha=0.7, edgecolor='k', marker='o', color=colors[i]
        )

    plt.title("t-SNE Visualization of Audio Features", fontsize=14)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="Languages", fontsize=10)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tsne_visualization_with_labels.png")
    plt.savefig(save_path)
    plt.show()
    print(f"t-SNE visualization saved to {save_path}")

tsne_visualization(features, labels, output_dir, label_names)
