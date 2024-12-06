import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 데이터 전처리 함수
def normalize(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)


# 데이터 로드 및 병합
def load_and_merge_data_from_dirs(base_dir):
    features = []
    labels = []
    label_mapping = {}

    for label_idx, language_dir in enumerate(sorted(os.listdir(base_dir))):
        lang_dir_path = os.path.join(base_dir, language_dir)
        if not os.path.isdir(lang_dir_path):
            continue
        label_mapping[language_dir] = label_idx
        mfcc_file = os.path.join(lang_dir_path, "mfccs.npy")
        if not os.path.exists(mfcc_file):
            continue
        mfcc_data = np.load(mfcc_file)
        mfcc_flat = mfcc_data.reshape(len(mfcc_data), -1)
        features.append(mfcc_flat)
        labels.extend([label_idx] * len(mfcc_data))

    features = np.vstack(features)
    labels = np.array(labels)
    return normalize(features), labels, label_mapping


# 하이퍼파라미터
num_batch = 64
base_dir = "/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/Train/numpyfiles/"
features, all_labels, label_mapping = load_and_merge_data_from_dirs(base_dir)
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(all_labels, dtype=torch.long)
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=num_batch, shuffle=True)

# 모델 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, z):
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.relu(self.bn2(self.fc2(h)))
        out = torch.sigmoid(self.fc_out(h))
        return out


# 손실 함수 정의
def reconstruction_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, reduction='mean')


def kl_divergence(mu, logvar):
    logvar = torch.clamp(logvar, min=-10, max=10)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def weights_init(m):
    if isinstance(m, nn.Linear):  # Linear 계층에 대해 He 초기화 적용
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):  # BatchNorm 계층 초기화
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# 모델 초기화
input_dim = features.shape[1]
hidden_dim = 128
latent_dim = 16
encoder = Encoder(input_dim, hidden_dim, latent_dim).to(device)
decoder = Decoder(latent_dim, hidden_dim, input_dim).to(device)

encoder.apply(weights_init)
decoder.apply(weights_init)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

# 학습 파라미터
num_epochs = 30
beta = 0.5  # KL 손실 가중치
log_dir = f"/hdd_ext/hdd3/joowoniese/languageRecognition/dataset/VAE_Model/VAEncoder_epoch{num_epochs}_batch{num_batch}"
os.makedirs(log_dir, exist_ok=True)

loss_log = []

# 학습 루프
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    train_loss = 0

    for batch in dataloader:
        data, _ = batch
        data = data.to(device)
        optimizer.zero_grad()

        mu, logvar = encoder(data)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        z = mu + eps * std
        recon_data = decoder(z)

        recon_loss = reconstruction_loss(recon_data, data)
        kl_loss = kl_divergence(mu, logvar)
        loss = recon_loss + beta * kl_loss

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    epoch_loss = train_loss / len(dataloader)
    loss_log.append({'epoch': epoch + 1, 'loss': epoch_loss})
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")

# 모델 저장
final_model_path = os.path.join(log_dir, "final_model.pth")
torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, final_model_path)
print(f"Final VAE model saved at: {final_model_path}")

# 손실 로그 저장
loss_df = pd.DataFrame(loss_log)
loss_csv_path = os.path.join(log_dir, f"VAE_loss_epoch{num_epochs}_batch{num_batch}.csv")
loss_df.to_csv(loss_csv_path, index=False)
print(f"Loss log saved to {loss_csv_path}")

# 손실 플롯 저장
plt.figure(figsize=(10, 6))
plt.plot(loss_df['epoch'], loss_df['loss'], marker='o', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Normalized)')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(log_dir, f"VAE_loss_plot_epoch{num_epochs}_batch{num_batch}.png")
plt.savefig(loss_plot_path)
plt.show()
print(f"Loss plot saved to {loss_plot_path}")

# Latent Space 시각화
encoder.eval()
all_latents = []
all_labels_list = []

with torch.no_grad():
    for data, labels in dataloader:
        data = data.to(device)
        mu, _ = encoder(data)
        all_latents.append(mu.cpu().numpy())
        all_labels_list.extend(labels.numpy())

all_latents = np.vstack(all_latents)
all_labels_array = np.array(all_labels_list)

# t-SNE 차원 축소
tsne = TSNE(n_components=2, random_state=42)
latents_2d = tsne.fit_transform(all_latents)

# 시각화
plt.figure(figsize=(10, 8))
for language_name, label_idx in label_mapping.items():
    plt.scatter(
        latents_2d[all_labels_array == label_idx, 0],
        latents_2d[all_labels_array == label_idx, 1],
        label=language_name,
        alpha=0.7,
        s=10
    )

plt.title("Latent Space Visualization with Labels")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.legend(title="Languages")
plt.grid(True)
latent_plot_path = os.path.join(log_dir, "latent_space_visualization_with_labels.png")
plt.savefig(latent_plot_path)
plt.show()
print(f"Latent space plot saved to {latent_plot_path}")
