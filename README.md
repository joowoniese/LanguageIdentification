# Language Identification
오직 음성 데이터를 사용해서 음성 특징을 포착하여 언어를 식별하는 모델

---

### 1. Virtual Environment Setup
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
### Language Classifier Test
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
