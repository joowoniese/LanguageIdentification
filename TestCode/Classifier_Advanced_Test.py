import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.functional import cosine_similarity

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 분류 모델 정의 (학습 코드와 동일)
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class LanguageDataset(Dataset):
    def __init__(self, wav2vec_features_path, encoder_features_dir, labels_path):
        # Wav2Vec2 특징 로드
        self.wav2vec_features = np.load(wav2vec_features_path)
        self.wav2vec_filenames = np.load(wav2vec_features_path.replace("audio_features.npy", "file_names.npy"))

        # 레이블 로드
        labels_df = pd.read_csv(labels_path)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(labels_df["Label"].unique()))}
        self.labels = labels_df["Label"].values
        self.filenames = labels_df["File"].values

        # VAE Latent 벡터 로드
        latent_vectors_path = os.path.join(encoder_features_dir, "latent_vectors.npy")
        file_names_path = os.path.join(encoder_features_dir, "file_names.npy")

        latent_file_names = np.load(os.path.join(encoder_features_dir, "file_names.npy"))
        wav2vec_file_names = np.load(wav2vec_features_path.replace("audio_features.npy", "file_names.npy"))

        # 교집합 확인
        common_files = set(latent_file_names) & set(wav2vec_file_names)
        print(f"Common files: {len(common_files)}")

        self.encoder_features = {}
        if os.path.exists(latent_vectors_path) and os.path.exists(file_names_path):
            latent_vectors = np.load(latent_vectors_path)
            latent_file_names = np.load(file_names_path)

            # 파일 이름과 VAE 벡터를 매핑
            self.encoder_features = {
                file_name: latent_vectors[idx]
                for idx, file_name in enumerate(latent_file_names)
            }

        # 매칭되지 않는 파일을 필터링
        self.valid_indices = [
            idx for idx, file_name in enumerate(self.filenames) if file_name in self.encoder_features
        ]
        self.filenames = self.filenames[self.valid_indices]
        self.labels = self.labels[self.valid_indices]

        # 선형 변환 정의
        self.encoder_linear = nn.Linear(16, 1024)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]

        # Wav2Vec2 특징 로드
        wav2vec_feature = self.wav2vec_features[np.where(self.wav2vec_filenames == file_name)[0][0]]
        wav2vec_feature = torch.tensor(wav2vec_feature, dtype=torch.float32)

        # VAE Latent 특징 로드 및 선형 변환
        encoder_feature = self.encoder_features[file_name]
        encoder_feature = torch.tensor(encoder_feature, dtype=torch.float32)
        encoder_feature = self.encoder_linear(encoder_feature)

        # 코사인 유사도 계산
        similarity = cosine_similarity(wav2vec_feature.unsqueeze(0), encoder_feature.unsqueeze(0))
        similarity = torch.tensor([similarity.item()], dtype=torch.float32)

        # 벡터 결합 + 코사인 유사도 추가
        combined_feature = torch.cat((wav2vec_feature, encoder_feature, similarity), dim=0)
        label = self.label_to_idx[self.labels[idx]]

        return combined_feature, torch.tensor(label, dtype=torch.long), file_name


# 데이터 경로 설정
wav2vec_features_path = "../dataset/Test/wav2vec_featuredata/audio_features.npy"
encoder_features_dir = "../dataset/Test/vae_latent/"
labels_path = "../dataset/Test/wav2vec_featuredata/audio_labels.csv"

# 테스트 데이터셋 및 데이터 로더 생성
test_dataset = LanguageDataset(wav2vec_features_path, encoder_features_dir, labels_path)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 로드
input_dim = next(iter(test_loader))[0].shape[1]
hidden_dim = 128
num_classes = len(test_dataset.label_to_idx)

classifier = Classifier(input_dim, hidden_dim, num_classes).to(device)
model_path = "../Models/Classifier/kfold_epoch50_batch32/classifier_model_fold3.pth"
classifier.load_state_dict(torch.load(model_path))
classifier.eval()

# 테스트 루프
print("Starting testing...")
test_loss = 0
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()

predictions = []
true_labels = []
file_names = []

with torch.no_grad():
    for inputs, targets, files in test_loader:  # 수정된 __getitem__으로 3개 값 반환
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = classifier(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(targets.cpu().numpy())
        file_names.extend(files)


# 레이블과 언어의 매핑 출력
print("\nLabel to Language Mapping:")
for label, idx in test_dataset.label_to_idx.items():
    print(f"{idx}: {label}")

# 샘플 50개 출력
sample_indices = random.sample(range(len(file_names)), 50)
print("\nPredictions (Random 50 Samples):")
print(f"{'File Name':<30}{'True Label':<15}{'Predicted Label':<15}")
for idx in sample_indices:
    true_label_name = list(test_dataset.label_to_idx.keys())[list(test_dataset.label_to_idx.values()).index(true_labels[idx])]
    predicted_label_name = list(test_dataset.label_to_idx.keys())[list(test_dataset.label_to_idx.values()).index(predictions[idx])]
    print(f"{file_names[idx]:<30}{true_label_name:<15}{predicted_label_name:<15}")

# 틀린 예측만 추출
incorrect_indices = [i for i in range(len(true_labels)) if true_labels[i] != predictions[i]]

# 틀린 예측에서 랜덤으로 50개 샘플링
num_samples = min(50, len(incorrect_indices))
sample_indices = random.sample(incorrect_indices, num_samples)

print("\nIncorrect Predictions (Random 50 Samples):")
print(f"{'File Name':<30}{'True Label':<15}{'Predicted Label':<15}")
for idx in sample_indices:
    true_label_name = list(test_dataset.label_to_idx.keys())[list(test_dataset.label_to_idx.values()).index(true_labels[idx])]
    predicted_label_name = list(test_dataset.label_to_idx.keys())[list(test_dataset.label_to_idx.values()).index(predictions[idx])]
    print(f"{file_names[idx]:<30}{true_label_name:<15}{predicted_label_name:<15}")


# 최종 테스트 성능 출력
test_acc = 100 * correct / total
test_loss /= len(test_loader)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
