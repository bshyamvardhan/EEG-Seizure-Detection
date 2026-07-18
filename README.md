# 🧠 EEG Seizure Detection using ML, CNN, Transfer Learning & XAI

## 📌 Overview

This project implements a **complete EEG classification pipeline** for epileptic seizure detection using:

- Machine Learning (SVM, Random Forest)
- Deep Learning (CNN on Spectrograms)
- Transfer Learning (ResNet50)
- Explainable AI (Grad-CAM)

The project classifies EEG signals into **Healthy, Interictal, and Ictal** classes using both traditional machine learning and deep learning techniques.

👉 Final performance reaches **~99% accuracy with interpretability.**

---

## 👥 Team Members

- **Boggarapu Shyam Vardhan**
- **Gunupati Venkata Bhuvana Mohan**

---

# 📂 Repository File Guide (IMPORTANT)

⚠️ All main code files are present in the **root directory**. Each file corresponds to a specific phase of the project.

---

## 🔹 Spectrogram Generation

**File:** `spectrograms.ipynb`

### Purpose

- Converts EEG `.txt` signals into spectrogram images
- Uses STFT with log scaling

### Output

```
spectrogram_dataset/
    healthy/
    ictal/
    interictal/
```

---

## 🔹 Phase 1 – Machine Learning

**File:** `Feature_Extraction/feature_extraction.py`

### Contains

- Signal preprocessing (Bandpass Filtering)
- Sub-band decomposition
- Feature extraction:
  - Statistical Features
  - Spectral Features
  - Entropy Features

### Output

- `EEG_Feature_Dataset.csv`

### Machine Learning Models

- Support Vector Machine (SVM)
- Random Forest
- Feature Selection

### Results

- Accuracy Scores
- Classification Report

---

## 🔹 Phase 2 – CNN (Spectrogram Classification)

**File:** `customCNN.ipynb`

### Contains

- ImageDataGenerator
- Data normalization (Mean & Standard Deviation)
- Custom CNN Architecture
- Ablation Study Variants

### Output

- Training Accuracy
- Validation Accuracy
- Training Curves

---

## 🔹 Phase 3 – Transfer Learning

**File:** `EEG_detection_Resnet_XAI.ipynb`

### Contains

- ResNet50 (ImageNet Pretrained)
- Fine-tuning
- Custom Classification Head

### Output

- Test Accuracy (~98–99%)
- Confusion Matrix

---

## 🔹 Explainable AI (Grad-CAM)

**File:** `EEG_detection_Resnet_XAI.ipynb`

### Contains

- Feature Map Extraction
- Gradient Computation
- Heatmap Generation

### Output

- Grad-CAM Overlay Images

---

## 📁 Results Folder

**Folder:** `Results/`

### Contains

- Confusion Matrices
- Accuracy Plots
- Grad-CAM Visualizations
- Model Comparison Graphs

👉 This folder provides a quick overview of the performance of all implemented models.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- OpenCV
- Matplotlib
- Librosa
- STFT
- Grad-CAM

---

## 📊 Dataset

The project uses EEG recordings for epileptic seizure detection. The signals are preprocessed and converted into spectrograms for deep learning models, while handcrafted features are extracted for machine learning models.

---

## 🚀 Models Implemented

- Support Vector Machine (SVM)
- Random Forest
- Custom CNN
- ResNet50 Transfer Learning
- Grad-CAM Explainable AI

---

## 📌 Project Outcome

This project demonstrates a complete EEG seizure detection pipeline using machine learning, deep learning, transfer learning, and explainable AI techniques. The combination of these approaches achieves high classification accuracy while providing visual explanations through Grad-CAM.
