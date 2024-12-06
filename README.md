# Automatic Spoken Language Identification Model (End-to-End Language Identification Model)
### A model designed to identify languages by capturing speech features solely from audio data
![Architecture](https://github.com/joowoniese/LanguageIdentification/blob/main/ModelInfo/language_fullarchitecture.drawio%20(4).png)
---

## Virtual Environment Setup
* create virtual envs
```
conda create -n langaugeIdentification python==3.9
```
* activate virtual envs
```
conda activate langaugeIdentification
```
* install packages/librarys
```
pip install -r requirements.txt
```
```
cd ./LanguageIdentification
```
## Model Test
### 1. Language Classifier Test
```
python ./TrainCode/Classifier_Advanced_Test.py
```


## Model Train
### 1. Run Pre-trained Wav2Vec 
```
python ./TrainCode/wav2vec.py
```
### 2. Train Customed VAE
```
python ./TrainCode/VAEncoder_MFCC_Advanced.py
```
### 3. Train Classifier VAE
```
python ./TrainCode/Classifier_Advanced.py
```



## Dataset Directory
- **wav2vec_featuredata/**: Contains Wav2Vec 2.0 extracted feature files (`audio_features.npy`, `file_names.npy`, etc.).
- **vae_latent/**: Latent vector files generated using a Variational Autoencoder (VAE).
- **spanish/**, **korean/**, **japanese/**, **french/**, **chinese/**: Language-specific directories containing raw audio files (`.wav` format).
- **numpyfiles/**: Stores additional preprocessed NumPy arrays for model input.
```
/LanguageIdentification/dataset/
├── Train/
│   ├── wav2vec_featuredata/
│   │   ├── audio_features.npy
│   │   ├── file_names.npy
│   │   └── audio_labels.csv
│   ├── vae_latent/
│   │   ├── latent_vectors.npy
│   │   └── file_names.npy
│   ├── MFCCs/
│   │   └── mfccs.npy
│   ├── spanish/
│   │   ├── file1.wav
│   │   └── file2.wav
│   ├── korean/
│   │   ├── file1.wav
│   │   └── file2.wav
│   ├── japanese/
│   │   ├── file1.wav
│   │   └── file2.wav
│   ├── french/
│   │   ├── file1.wav
│   │   └── file2.wav
│   └── chinese/
│       ├── file1.wav
│       └── file2.wav
├── Test/
│   ├── wav2vec_featuredata/
│   ├── vae_latent/
│   ├── MFCCs/
│   ├── spanish/
│   ├── korean/
│   ├── japanese/
│   ├── french/
│   └── chinese/
└── classifier_Model/
    ├── kfold_epoch50_batch32/
    │   ├── classifier_model_fold1.pth
    │   ├── classifier_model_fold2.pth
    │   ├── classifier_model_fold3.pth
    │   └── training_results.png
    └── other_logs/
        └── example_log.txt
```
---
### Dataset download
Korean - [Korean Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)

Chinese - [Chinese Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/chinese-single-speaker-speech-dataset)

Japanese - [Japanese Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/japanese-single-speaker-speech-dataset)

Spanish - [Spanish Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/spanish-single-speaker-speech-dataset)

French - [French Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/french-single-speaker-speech-dataset)

* Dowmload code
```
import kagglehub
import shutil
import os

dataset_id = "bryanpark/chinese-single-speaker-speech-dataset"
default_path = kagglehub.dataset_download(dataset_id)

target_dir = "/hdd_ext/hdd3/joowoniese/languageRecognition/"

if os.path.exists(default_path):
    shutil.move(default_path, target_dir)

print(f"Dataset moved to: {target_dir}")
```

