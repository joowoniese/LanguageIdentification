import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn.functional import cosine_similarity

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터 설정
num_epochs = 50
num_batchs = 32
k_folds = 3  # K-Fold의 fold 수
save_dir = f"/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/classifier_Model/kfold_epoch{num_epochs}_batch{num_batchs}"
os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)

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

        return combined_feature, torch.tensor(label, dtype=torch.long)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# Focal Loss 정의 (클래스별 알파 적용)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 클래스별 가중치 (리스트 형태)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 정답 클래스의 예측 확률
        if self.alpha is not None:
            # 클래스별 가중치 적용
            alpha = torch.tensor(self.alpha).to(inputs.device)
            alpha_t = alpha[targets]  # 정답 클래스에 해당하는 \(\alpha\) 값 선택
            BCE_loss = alpha_t * BCE_loss
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# 데이터 비율 기반으로 \(\alpha\) 계산
def calculate_alpha(labels):
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    alpha = [total_samples / class_counts[cls] for cls in sorted(class_counts.keys())]

    # 정규화 (선택 사항: 모든 \(\alpha\) 값의 합을 1로 만듦)
    alpha = [a / sum(alpha) for a in alpha]
    return alpha

# 데이터 경로 설정
wav2vec_features_path = "/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/Train/wav2vec_featuredata/audio_features.npy"
encoder_features_dir = "/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/Train/vae_latent/"
labels_path = "/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/Train/wav2vec_featuredata/audio_labels.csv"

# 데이터셋 생성
dataset = LanguageDataset(wav2vec_features_path, encoder_features_dir, labels_path)

# K-Fold 교차 검증
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Fold 결과 저장
fold_results = {
    "fold": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

# K-Fold 교차 검증
for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), dataset.labels)):
    print(f"\nStarting Fold {fold + 1}/{k_folds}...")

    # Train/Validation Split
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=num_batchs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=num_batchs, shuffle=False)

    # 모델 초기화
    input_dim = next(iter(train_loader))[0].shape[1]
    hidden_dim = 128
    num_classes = len(dataset.label_to_idx)
    classifier = Classifier(input_dim, hidden_dim, num_classes).to(device)
    initialize_weights(classifier)

    # 클래스별 알파 계산
    train_labels = [dataset.labels[i] for i in train_idx]
    val_labels = [dataset.labels[i] for i in val_idx]
    alpha = calculate_alpha(train_labels)  # 클래스별 \(\alpha\) 값 계산
    print(f"Fold {fold + 1} Alpha values: {alpha}")
    print(f"Fold {fold + 1}:")
    print(f"  Train class distribution: {Counter(train_labels)}")
    print(f"  Validation class distribution: {Counter(val_labels)}")

    # 손실 함수 및 옵티마이저
    criterion = FocalLoss(alpha=alpha, gamma=2)  # 클래스별 \(\alpha\) 적용
    optimizer = torch.optim.Adam(classifier.parameters(), lr=2e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # 학습
    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_acc = 100 * correct / total

        # Validation
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = classifier(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_acc = 100 * correct / total

        print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

    # Fold 결과 저장
    fold_results["fold"].append(fold + 1)
    fold_results["train_loss"].append(train_loss / len(train_loader))
    fold_results["val_loss"].append(val_loss / len(val_loader))
    fold_results["train_acc"].append(train_acc)
    fold_results["val_acc"].append(val_acc)

    # Fold별 모델 저장
    fold_model_path = os.path.join(save_dir, f"classifier_model_fold{fold + 1}.pth")
    torch.save(classifier.state_dict(), fold_model_path)
    print(fold_model_path)

# Fold 결과를 데이터프레임으로 변환
df_results = pd.DataFrame(fold_results)

# 정확도 및 손실 그래프 그리기 함수 추가
def plot_training_results(df_results, save_dir):
    plt.figure(figsize=(12, 6))

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(df_results['fold'], df_results['train_acc'], marker='o', label='Train Accuracy')
    plt.plot(df_results['fold'], df_results['val_acc'], marker='o', label='Validation Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Fold')
    plt.legend()
    plt.grid(True)

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(df_results['fold'], df_results['train_loss'], marker='o', label='Train Loss')
    plt.plot(df_results['fold'], df_results['val_loss'], marker='o', label='Validation Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.title('Loss per Fold')
    plt.legend()
    plt.grid(True)

    # 그래프 저장 및 출력
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_results.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Training results plot saved to {save_path}")

# 그래프 저장 경로 설정
plot_training_results(df_results, save_dir)