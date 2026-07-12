# Multilingual Mobility: Audio-Based Language ID for Automotive Systems 🚗🎙️

[![Paper](https://img.shields.io/badge/Journal-Applied%20Sciences%20(2025)-blue.svg)](https://doi.org/10.3390/app15169209)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat-flat&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-Embedded-EE4C2C.svg?style=flat-flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of the paper: **"Multilingual Mobility: Audio-Based Language ID for Automotive Systems"** (*Applied Sciences*, 2025).

This repository introduces an end-to-end, noise-robust **Automatic Spoken Language Identification (LID)** framework designed for next-generation intelligent in-vehicle infotainment (IVI) systems. By combining self-supervised contextual representations with generative latent modeling, the system achieves seamless multilingual voice interaction across five target languages (**Korean, Japanese, Chinese, Spanish, and French**) without requiring manual language selection or safe-distracting profile setups.

---

## 📌 Project Overview & Automotive Challenges
Operating a voice assistant inside a moving vehicle introduces unique acoustic limitations—specifically **engine hums, road vibration, tire noise, and cabin reverberation**—which severely degrade standard speech classification metrics[cite: 3]. Furthermore, multi-passenger scenarios can cause acoustic overlapping[cite: 3].

To address these vulnerabilities, this framework rejects heavy, power-hungry transformer architectures in favor of a **lightweight Dual-Pipeline Feature Fusion network** tailored for resource-constrained automotive edge processors (e.g., NVIDIA Jetson, Raspberry Pi)[cite: 3]. 

### Key Innovations
- **Dual-Representation Encoding:** Combines global high-level phonetic contexts from a pre-trained `Wav2Vec 2.0` architecture with local speaker-agnostic generative distributions captured by a customized `Variational Autoencoder (VAE)`[cite: 3].
- **Acoustic Interaction via Cosine Similarity:** Computes directional correlation maps between divergent embedding sources to enhance final dense classification[cite: 3].
- **Noise-Insulated Data Profile:** Reconstructed using strict vehicular engine and traffic noise augmentations sourced from AI Hub datasets[cite: 3].

---

## 🛠️ System Architecture

The pipeline processes raw audio frames directly, decoupling low-level spectro-temporal profiles and global context chains into localized matrices before mapping variables onto an MLP classification head[cite: 3].

<p align="center">
  <img src="https://github.com/joowoniese/LanguageIdentification/blob/main/ModelInfo/language_fullarchitecture.drawio%20(4).png" width="60%" alt="System Architecture Overview" />
</p>

1. **Wav2Vec 2.0 Branch:** Feeds raw signal matrices through temporal self-attention masks to extract structured contextual sound shapes[cite: 3].
2. **Custom VAE Branch:** Maps log-scale **MFCC** inputs onto a regularized distribution vector to preserve accent variations, intonations, and articulation layouts despite background in-cabin chatter[cite: 3].
3. **Similarity-Gated MLP Classifier:** Extends VAE vectors via dense projections, extracts mutual cosine alignments, and feeds the merged $2049\text{-dimensional}$ space into an optimized classifier wrapped with Hard-Sample Focal Loss criteria[cite: 3].

---

## 📊 Experimental & Benchmarking Results

### 1. Embedded Optimization & Real-Time Performance
Evaluated across production edge platforms representing in-car hardware arrays[cite: 3]:

| Compute Platform / Environment | Computational Cost | Average Inference Latency | Peak Memory Usage |
| :--- | :---: | :---: | :---: |
| NVIDIA RTX 8000 GPU | 0.27 MFLOPs[cite: 3] | **0.344 ms**[cite: 3] | 11.47 MB[cite: 3] |
| NVIDIA Jetson Nano (TensorRT) | 0.19 MFLOPs[cite: 3] | **3.5 ms**[cite: 3] | **9.0 MB**[cite: 3] |
| Raspberry Pi 5 (CPU Only) | 0.27 MFLOPs[cite: 3] | 15.0 ms[cite: 3] | 15.0 MB[cite: 3] |

### 2. Comparative Resource Analysis against SOTA Baselines
The framework balances state-of-the-art accuracy with a miniature hardware footprint crucial for edge mobility[cite: 3]:

| Evaluation Model Index | Korean (F1) | Spanish (F1) | Memory Allocation | Inference Latency |
| :--- | :---: | :---: | :---: | :---: |
| Traditional Audio CNN [44] | 0.50[cite: 3] | 0.67[cite: 3] | 90.58 MB[cite: 3] | 550.5 ms[cite: 3] |
| OpenAI Whisper [45] | 0.99[cite: 3] | 0.99[cite: 3] | 2989.86 MB[cite: 3] | 193.5 ms[cite: 3] |
| SpeechBrain Baseline [46] | 0.99[cite: 3] | 0.26[cite: 3] | 191.84 MB[cite: 3] | 88.4 ms[cite: 3] |
| **Our Proposed Framework** | **1.00**[cite: 3] | **0.99**[cite: 3] | **11.47 MB**[cite: 3] | **0.344 ms**[cite: 3] |

---

## 🚀 Virtual Environment Setup

### Conda Installation
```bash
# clone project
git clone https://github.com/joowoniese/LanguageIdentification.git
cd LanguageIdentification

# create conda environment
conda create -n langaugeIdentification python==3.9
conda activate langaugeIdentification

# install packages/librarys
pip install -r requirements.txt
```

---

## 🗂️ Dataset Directory Structure

Download the Preprocessed Training Vectors and map files within your active directory as detailed below:
```bash
/LanguageIdentification/dataset/
├── Train/
│   ├── wav2vec_featuredata/       # Extracted self-supervised features
│   │   ├── audio_features.npy
│   │   ├── file_names.npy
│   │   └── audio_labels.csv
│   ├── vae_latent/                # Encoded VAE target properties
│   │   ├── latent_vectors.npy
│   │   └── file_names.npy
│   ├── MFCCs/                     # Ground acoustic feature arrays
│   │   └── mfccs.npy
│   ├── spanish/                   # Raw audio archives (.wav)
│   ├── korean/
│   ├── japanese/
│   ├── french/
│   └── chinese/ (Augmented)
└── classifier_Model/
    └── kfold_epoch50_batch32/     # Evaluated 5-Fold cross validation weights
        ├── classifier_model_fold1.pth
        ├── classifier_model_fold2.pth
        └── training_results.png
```

---

#### 🗂️ Citation

Oh, Joowon and Lee, Jeaho.
"Multilingual Mobility: Audio-Based Language ID for Automotive Systems." Applied Sciences, vol. 15, no. 16, 2025, p. 9209.
@article{oh2025multilingual,
title={Multilingual Mobility: Audio-Based Language ID for Automotive Systems},
author={Oh, Joowon and Lee, Jeaho},
journal={Applied Sciences},
volume={15},
number={16},
pages={9209},
year={2025},
publisher={MDPI}
}
