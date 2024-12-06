# Language Identification
오직 음성 데이터를 사용해서 음성 특징을 포착하여 언어를 식별하는 모델

## Model Training
```
pip install -r requirements.txt
```

## Model Test
```

```


### Description of Each Folder
- **wav2vec_featuredata/**: Contains Wav2Vec 2.0 extracted feature files (`audio_features.npy`, `file_names.npy`, etc.).
- **vae_latent/**: Latent vector files generated using a Variational Autoencoder (VAE).
- **spanish/**, **korean/**, **japanese/**, **french/**, **chinese/**: Language-specific directories containing raw audio files (`.wav` format).
- **numpyfiles/**: Stores additional preprocessed NumPy arrays for model input.

This structure ensures the data is well-organized for language recognition tasks.


---

### **Key Notes**
- **Top-level folders**:
  - `Train/`: Contains all training data, features, and processed files.
  - `Test/`: Mirrors `Train/` structure for testing.
  - `classifier_Model/`: Stores trained models, logs, and results.

- **Feature directories**:
  - `wav2vec_featuredata/`: Wav2Vec 2.0 extracted features.
  - `vae_latent/`: Latent vectors from a VAE model.

- **Language folders**:
  - Separate folders (`spanish`, `korean`, `japanese`, `french`, `chinese`) for raw audio files by language.

---


## Dataset
Korean - [Korean Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)

Chinese - [Chinese Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/chinese-single-speaker-speech-dataset)

Japanese - [Japanese Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/japanese-single-speaker-speech-dataset)

Spanish - [Spanish Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/spanish-single-speaker-speech-dataset)

French - [French Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/french-single-speaker-speech-dataset)
