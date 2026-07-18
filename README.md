
Epileptic-Seizure-Detection-EEG
Repository navigation
Code
Issues
Pull requests
Actions
End-to-end EEG seizure detection system using Machine Learning, CNN, Transfer Learning, and Explainable AI (Grad-CAM) for accurate and interpretable classification of brain states.

 0 stars
 0 forks
 0 watching
 1 Branch
 0 Tags
 Activity
Public repository
BhuvanMohan2005/Epileptic-Seizure-Detection-EEG
Name	
BhuvanMohan2005
BhuvanMohan2005
2 months ago
Base_Dataset
2 months ago
Feature_Extraction
2 months ago
Results
2 months ago
spectrogram_dataset
2 months ago
.gitattributes
2 months ago
.gitignore
2 months ago
EEG_detection_Resnet_XAI.ipynb
2 months ago
EEG_type_detection.ipynb
2 months ago
ML_analysis.ipynb
2 months ago
README.md
2 months ago
Repository files navigation
README
🧠 EEG Seizure Detection using ML, CNN, Transfer Learning & XAI
📌 Overview
This project implements a complete EEG classification pipeline for epileptic seizure detection using:

Machine Learning (SVM, Random Forest)
Deep Learning (CNN on spectrograms)
Transfer Learning (ResNet50)
Explainable AI (Grad-CAM)
👉 Final performance reaches ~99% accuracy with interpretability

📂 Repository File Guide (IMPORTANT)
⚠️ All main code files are present in the root directory itself. Each file corresponds to a specific phase of the project.

🔹 Spectrogram Generation
<spectrograms.ipynb>
Purpose:
Converts EEG .txt signals → spectrogram images
Uses STFT + log scaling
👉 Output:

spectrogram_dataset/
    healthy/
    ictal/
    interictal/
🔹 Phase 1 – Machine Learning
<Feature_Extraction/feature_extraction.py>
Contains:
Signal preprocessing (bandpass filtering)

Sub-band decomposition

Feature extraction:

Statistical
Spectral
Entropy
👉 Generates:

EEG_Feature_Dataset.csv
Contains:
SVM (RBF)
Random Forest
Feature selection
👉 Output:

Accuracy scores
Classification report
🔹 Phase 2 – CNN (Spectrogram Classification)
<customCNN.ipynb>
Contains:
Data loading using ImageDataGenerator
Normalization (mean/std)
Custom CNN architecture
Ablation study variants
👉 Output:

Model accuracy
Training curves
🔹 Phase 3 – Transfer Learning
<EEG_detection_Resnet_XAI.ipynb>
Contains:
ResNet50 (ImageNet pretrained)
Fine-tuning
Custom classification head
👉 Output:

Test accuracy (~98–99%)
Confusion matrix
🔹 Explainable AI (Grad-CAM)
<EEG_detection_Resnet_XAI.ipynb>
Contains:
Feature map extraction
Gradient computation
Heatmap generation
👉 Output:

Grad-CAM overlay images
📁 Results Folder
Results/
Contains:
Confusion matrices
Accuracy plots
Grad-CAM outputs
Model comparison visuals
👉 This is the best place to understand model performance quickly

▶️ Execution Flow (VERY IMPORTANT)
Follow this order to reproduce the project:

1. Run spectrogram generation file
2. Run Phase 1 (feature extraction + ML)
3. Run Phase 2 (CNN training)
4. Run Phase 3 (transfer learning)
5. Run Grad-CAM for visualization
🔬 Methodology Summary
EEG → Preprocessing → 
    ├── Feature Extraction → ML (SVM, RF)
    ├── Spectrogram → CNN
    └── Transfer Learning (ResNet50) → Grad-CAM
#Main files are organized based on project phases:

Phase 1 (Machine Learning):

Feature extraction and ML models (SVM, Random Forest)
Phase 2 (CNN):

Spectrogram generation and CNN training
Phase 3 (Transfer Learning + XAI):

ResNet50 model and Grad-CAM visualization
Results:

Contains outputs, confusion matrices, and Grad-CAM images
📊 Results Summary
Model	Accuracy	Notes
SVM	94%	Baseline
Random Forest	97%	Best ML model
CNN (Dropout)	98%	Strong generalization
ResNet50	98–99%	Best overall
EfficientNet	68%	Poor adaptation
GoogleNet	89%	Not Good Enough
🔍 Explainability (Grad-CAM)
Grad-CAM visualizations show:

Healthy → stable patterns
Ictal → high-energy spikes
Interictal → mixed patterns
👉 Confirms model focuses on meaningful EEG regions

🧠 Key Contributions
End-to-end EEG classification pipeline
Comparison: ML vs CNN vs Transfer Learning
High accuracy (~99%)
Explainable AI integration
⚠️ Notes
Code is organized phase-wise logically, not folder-wise
Results are available in the Results folder
Each file corresponds to a specific stage
🧠 Key Insights
Feature-based Machine Learning provides a strong baseline for EEG classification
CNN models eliminate the need for manual feature extraction
Transfer learning improves generalization and reduces training complexity
Explainable AI (Grad-CAM) enhances model interpretability and trustworthiness
⚠️ Limitations
Dataset size is limited (Bonn EEG dataset)
No temporal sequence modeling (e.g., LSTM or Transformers not used)
No real-time deployment or streaming EEG analysis
🔮 Future Work
Hybrid CNN + LSTM or Transformer-based architectures for temporal learning
Real-time EEG seizure detection system
Evaluation on large multi-patient clinical datasets
Clinical validation for hospital deployment
🛠️ Technologies Used
Python
NumPy, SciPy
Matplotlib
Scikit-learn
TensorFlow / Keras
▶️ How to Run
Generate spectrogram dataset from EEG signals
Train Phase 1: Machine Learning models (SVM, Random Forest)
Train Phase 2: CNN model on spectrogram images
Train Phase 3: Transfer Learning (ResNet50 fine-tuning)
Apply Grad-CAM for model explainability and visualization
📌 Conclusion
This project presents a complete EEG-based seizure detection pipeline combining:

Machine Learning
Deep Learning
Transfer Learning
Explainable AI (XAI)
The proposed system achieves high classification accuracy (~99%) while maintaining interpretability, making it suitable for real-world medical AI applications.

