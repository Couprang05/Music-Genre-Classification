
---

#  Audio Signal Processing: Comparative Study of Machine Learning Algorithms for Music Genre Classification

<p align="center">

<img src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
<img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg" />
<img src="https://img.shields.io/badge/Dataset-GTZAN-green.svg" />
<img src="https://img.shields.io/badge/Accuracy-76%25-brightgreen.svg" />
<img src="https://img.shields.io/badge/Status-Completed-success.svg" />
<img src="https://img.shields.io/badge/Domain-Audio%20Signal%20Processing-purple.svg" />

</p>

---

##  Overview

Music genre classification is a key task in **audio signal processing**, enabling applications such as music recommendation systems, automated tagging, and intelligent media organization.

This project presents a **comparative study of six machine learning algorithms** to classify music genres using engineered audio features extracted from the GTZAN dataset.

Instead of raw audio, signal processing features such as **MFCC, chroma, spectral contrast, and zero-crossing rate** are used to efficiently represent musical characteristics.

The project implements a complete **end-to-end ML pipeline**, including preprocessing, training, hyperparameter tuning, evaluation, and performance comparison.

---

##  Objectives

* Build a complete ML pipeline for music genre classification
* Perform audio feature preprocessing and normalization
* Train and compare multiple machine learning models
* Optimize models using GridSearchCV and cross-validation
* Evaluate models using multiple performance metrics
* Identify the best-performing algorithm

---

##  Dataset

**Dataset:** GTZAN Music Genre Dataset
🔗 [https://www.kaggle.com/datasets/ggalok/music-genre-classification](https://www.kaggle.com/datasets/ggalok/music-genre-classification)

### Dataset Details

| Property          | Value      |
| ----------------- | ---------- |
| Total audio files | 1000       |
| Genres            | 10         |
| Audio length      | 30 seconds |
| Sampling rate     | 22050 Hz   |
| Format            | WAV        |

### Genres

Blues • Classical • Country • Disco • Hiphop
Jazz • Metal • Pop • Reggae • Rock

---

##  Audio Feature Engineering

The following signal processing features were used:

### MFCC (Mel-Frequency Cepstral Coefficients)

Captures timbral and perceptual sound characteristics.

### Chroma Features

Represents pitch and harmonic content.

### Spectral Contrast

Measures spectral peaks vs valleys.

### Zero Crossing Rate

Captures rhythm and noisiness of signal.

---

##  Machine Learning Pipeline

```
Audio Dataset
     ↓
Feature Extraction
     ↓
Data Preprocessing
     ↓
Feature Scaling
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Performance Comparison
```

---

##  Machine Learning Models Used

| Model               | Description                     |
| ------------------- | ------------------------------- |
| KNN                 | Distance-based classification   |
| Logistic Regression | Linear probabilistic classifier |
| Decision Tree       | Tree-based classification       |
| Naive Bayes         | Probabilistic classifier        |
| SVM                 | Maximum margin classifier       |
| Random Forest       | Ensemble learning classifier    |

---

##  Hyperparameter Tuning

Used:

* GridSearchCV
* 5-Fold Cross Validation
* Accuracy-based optimization

This improves model performance and prevents overfitting.

---

##  Results

### Model Performance Comparison

| Model               | Accuracy | Precision | Recall   | F1 Score |
| ------------------- | -------- | --------- | -------- | -------- |
| KNN                 | 65%      | 0.65      | 0.64     | 0.64     |
| Logistic Regression | 71%      | 0.73      | 0.73     | 0.72     |
| Decision Tree       | 62%      | 0.63      | 0.63     | 0.62     |
| Naive Bayes         | 50%      | 0.58      | 0.53     | 0.51     |
| SVM                 | 74%      | 0.74      | 0.75     | 0.74     |
| Random Forest       | **76%**  | **0.77**  | **0.78** | **0.77** |

---

##  Best Model: Random Forest

**Why Random Forest performed best:**

✔ Ensemble learning improves accuracy

✔ Reduces overfitting

✔ Handles high-dimensional features well

✔ Robust and stable predictions

---

##  Project Structure

```
Music-Genre-Classification/
│
├── Data/
│   └── features_30_sec.csv
│
├── models/
│   ├── KNN_mod.pkl
│   ├── Logistic_Regression_mod.pkl
│   ├── Decision_Tree_mod.pkl
│   ├── Naive_Bayes_mod.pkl
│   ├── SVM_mod.pkl
│   └── Random_Forest_mod.pkl
│
├── prepro_mod.py
├── mod_train.py
├── mod_eval.py
│
└── README.md
```

---

##  Technologies Used

* Python
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Joblib

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  How to Run

### Train models

```bash
python mod_train.py
```

### Evaluate models

```bash
python mod_eval.py
```

---

##  Output

The system generates:

* Model accuracy comparison
* Confusion matrices
* Precision, recall, F1 score plots
* Saved trained models

---

##  Applications

* Music recommendation systems
* Automated music tagging
* Streaming platform categorization
* Audio content analysis
* Intelligent music libraries

---

##  Future Improvements

* Deep Learning using CNN on spectrograms
* Feature dimensionality reduction (PCA)
* Real-time genre classification
* Deploy using Streamlit or Flask
* Train on larger datasets

---

##  Author

**Harshita Bhatnagar**

BTech. CSE (Spec. AIML)


---
