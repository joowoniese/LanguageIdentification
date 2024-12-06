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

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face 캐시 경로 설정
os.environ["HF_HOME"] = "/home/joowoniese/huggingface_cache"

# 모델 이름 설정
model_name = "facebook/wav2vec2-large-xlsr-53"

# Feature Extractor 및 모델 로드
print("Loading feature extractor and model...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).to(device)

# 데이터 디렉토리 설정 (디렉토리별로 레이블 지정)
data_dir = "/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/Train/"
sub_dirs = [os.path.join(data_dir, sub_dir) for sub_dir in os.listdir(data_dir)]

# 파일과 레이블을 저장할 리스트
audio_files = []
labels = []

# 각 디렉토리의 파일과 레이블 읽기
for sub_dir in sub_dirs:
    label = os.path.basename(sub_dir)  # 디렉토리 이름이 레이블
    files = glob.glob(os.path.join(sub_dir, "*.wav"))
    audio_files.extend(files)
    labels.extend([label] * len(files))

print(f"Found {len(audio_files)} audio files across {len(sub_dirs)} labels: {set(labels)}")

# 오디오 데이터 처리 및 특징 추출
features_list = []
file_names_list = []  # 파일 이름 저장
labels_list = []

for i, file_path in enumerate(audio_files, start=1):
    try:
        # 파일 이름과 레이블 저장
        file_name = os.path.basename(file_path)
        label = labels[i - 1]

        print(f"[{i}/{len(audio_files)}] Processing file: {file_name} (Label: {label})")

        # 음성 파일 로드
        waveform, sample_rate = torchaudio.load(file_path)

        # 오디오를 16kHz로 리샘플링
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

        # 스테레오를 모노로 변환
        waveform = waveform.mean(dim=0)  # (samples,)

        # 차원 추가: 모델 입력은 (batch_size, sequence_length) 필요
        waveform = waveform.unsqueeze(0)  # (1, samples)

        # Wav2Vec 2.0을 위한 전처리 (Feature Extractor 사용)
        inputs = feature_extractor(waveform.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)

        # 입력 텐서를 GPU로 이동
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 모델 예측
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        # 중간 출력 (특징 벡터)
        hidden_states = outputs.last_hidden_state

        # 특징 벡터 추출
        features = hidden_states.mean(dim=1).squeeze()
        features_list.append(features.cpu().numpy())
        file_names_list.append(file_name)  # 파일 이름 추가
        labels_list.append(label)

        print(f"[{i}/{len(audio_files)}] Successfully processed file: {file_name}")

    except Exception as e:
        print(f"[{i}/{len(audio_files)}] Error processing {file_name}: {e}")

# 벡터화 결과 및 레이블 저장
print("Saving features, labels, and file names...")
output_dir = "/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/Train/wav2vec_featuredata/"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "audio_features.npy"), np.array(features_list))
np.save(os.path.join(output_dir, "file_names.npy"), np.array(file_names_list))  # 파일 이름 저장
pd.DataFrame({"File": file_names_list, "Label": labels_list}).to_csv(os.path.join(output_dir, "audio_labels.csv"), index=False)

print(f"Features saved to {os.path.join(output_dir, 'audio_features.npy')}")
print(f"File names saved to {os.path.join(output_dir, 'file_names.npy')}")
print(f"Labels saved to {os.path.join(output_dir, 'audio_labels.csv')}")

# 특징 벡터 및 레이블 로드
features = np.load(os.path.join(output_dir, "audio_features.npy"))
file_names = np.load(os.path.join(output_dir, "file_names.npy"))
labels = pd.read_csv(os.path.join(output_dir, "audio_labels.csv"))["Label"].values


# 레이블 번호와 언어 이름 매핑 (문자열 레이블)
label_names = {
    "chinese": "Chinese",
    "japanese": "Japanese",
    "french": "French",
    "spanish": "Spanish",
    "korean": "Korean"
}

def tsne_visualization(features, labels, output_dir, label_names):
    """
    t-SNE를 사용한 특징 벡터 시각화.

    Args:
        features (np.ndarray): 고차원 특징 벡터.
        labels (np.ndarray): 데이터의 레이블 배열 (문자열).
        output_dir (str): 시각화 결과 저장 디렉토리.
        label_names (dict): 레이블 문자열과 이름의 매핑 (e.g., {"chinese": "Chinese", ...}).
    """
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # tab10 컬러맵

    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)
        plt.scatter(
            tsne_features[idx, 0], tsne_features[idx, 1],
            label=label_names[label],  # 레이블 이름 사용
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

# 호출 예시
tsne_visualization(features, labels, output_dir, label_names)