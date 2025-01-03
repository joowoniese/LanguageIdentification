import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#  wav2vec_vocoder import features

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def normalize(data):
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

def load_and_merge_data_from_dirs(base_dir):
    features = []
    labels = []
    file_names = []
    label_mapping = {}

    for label_idx, language_dir in enumerate(sorted(os.listdir(base_dir))):
        lang_dir_path = os.path.join(base_dir, language_dir)
        if not os.path.isdir(lang_dir_path):
            continue
        label_mapping[language_dir] = label_idx
        mfcc_file = os.path.join(lang_dir_path, "mfccs.npy")
        file_names_file = os.path.join(lang_dir_path, "file_names.npy")
        if not (os.path.exists(mfcc_file) and os.path.exists(file_names_file)):
            continue
        mfcc_data = np.load(mfcc_file)
        file_names.extend(np.load(file_names_file))
        mfcc_flat = mfcc_data.reshape(len(mfcc_data), -1)
        features.append(mfcc_flat)
        labels.extend([label_idx] * len(mfcc_data))

    features = np.vstack(features)
    labels = np.array(labels)
    return normalize(features), labels, np.array(file_names), label_mapping

def load_test_data(test_dir, batch_size=32):
    features, labels, file_names, label_mapping = load_and_merge_data_from_dirs(test_dir)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, features, labels, file_names, label_mapping


def load_encoder(model_path, input_dim, hidden_dim, latent_dim, device):
    encoder = Encoder(input_dim, hidden_dim, latent_dim).to(device)
    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()
    return encoder

def extract_latent_space(encoder, test_dataloader, save_latent_dir, file_names, labels, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("Extracting Latent Space...")
    latent_space = []
    with torch.no_grad():
        for batch in test_dataloader:
            data, _ = batch
            data = data.to(device)
            mu, _ = encoder(data)  # Latent Space의 평균(mu)만 추출
            latent_space.append(mu.cpu().numpy())

    # Flatten the latent space features
    latent_space = np.vstack(latent_space)

    os.makedirs(save_latent_dir, exist_ok=True)
    np.save(os.path.join(save_latent_dir, "latent_vectors.npy"), latent_space)
    np.save(os.path.join(save_latent_dir, "true_labels.npy"), labels)
    np.save(os.path.join(save_latent_dir, "file_names.npy"), file_names)
    print(f"Latent vectors, labels, and file names saved to {save_latent_dir}")

    return latent_space

def visualize_latent_space(latent_space, labels, save_latent_dir, label_mapping):
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latent_space)

    plt.figure(figsize=(10, 8))
    for label_name, label_idx in label_mapping.items():
        indices = labels == label_idx
        plt.scatter(
            latents_2d[indices, 0],
            latents_2d[indices, 1],
            label=label_name,
            alpha=0.7,
            s=10
        )

    plt.title("Latent Space Visualization with Labels")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend(title="Languages")
    plt.grid(True)
    save_path = os.path.join(save_latent_dir, "latent_space_visualization_with_labels.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Latent space plot saved to {save_path}")

if __name__ == "__main__":
    model_path = "../Models/VAE/VAEncoder_epoch30_batch64/final_model.pth"
    test_dir = "../dataset/Test/MFCCs/"
    save_latent_dir = "../dataset/Test/vae_latent/"

    batch_size = 32

    test_dataloader, features, labels, file_names, label_mapping = load_test_data(test_dir, batch_size)
    input_dim = features.shape[1]  
    hidden_dim = 128
    latent_dim = 16
    encoder = load_encoder(model_path, input_dim, hidden_dim, latent_dim, device="cuda")

    latent_space = extract_latent_space(encoder, test_dataloader, save_latent_dir, file_names, labels, device="cuda")

    visualize_latent_space(latent_space, labels, save_latent_dir, label_mapping)
