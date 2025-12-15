# Face Recognition Pipeline using ArcFace

An end-to-end face recognition system built on **ArcFace embeddings**, designed for clustering unknown identities, training multiple classifiers, and generating **Kaggle-ready submission files**.

The project is structured as a **modular pipeline**, where each processing step is implemented in a separate script for clarity, debugging, and experimentation.

---

## Features

- Face detection with **RetinaFace**
- Face alignment to **112×112**
- Feature extraction using **ArcFace**
- Unsupervised clustering with **KMeans**
- Cluster validation using **silhouette scores**
- Multiple classification models:
  - Centroid (cosine similarity)
  - k-Nearest Neighbors
  - Support Vector Machine (linear)
  - Logistic Regression
- Automatic **Kaggle submission generation**
- Support for **multiple faces per image**
- HEIC → JPG conversion

---

## Project Structure

```
.
├── trainset/                 # Training images (multiple faces per image)
├── testset/                  # Test images (Kaggle)
├── cropped_faces/            # Aligned faces (112x112)
│   └── faces_metadata.csv    # Face detection metadata
├── clusters/                 # KMeans clusters (one folder per identity)
├── models/                   # Trained classifiers
├── submission/               # Kaggle submission CSV files
├── k_plots/                  # Elbow & PCA plots
├── silhouette_plots/         # Silhouette analysis plots
│
├── 1B_DetectFaces.py
├── 2B_CreateEmbeddings.py
├── 3B_Global_K.py
├── 4B_Clustering.py
├── 5B_SilhouetteScore.py
├── 6B_TrainClassifier.py
├── 7B_GenerateSubmission.py
├── requirements.txt
└── README.md
```

---

## 🔍 Pipeline Overview

### 1️⃣ Face Detection & Alignment
**Script:** `1B_DetectFaces.py`

- Face detection using **RetinaFace**
- Filters detections based on confidence and minimum size
- Aligns faces to **112×112**
- Saves aligned faces and metadata

---

### 2️⃣ Face Embedding Extraction
**Script:** `2B_CreateEmbeddings.py`

- Uses **ArcFace** for feature extraction
- Generates L2-normalized embeddings
- Stores embeddings in `parquet` format

---

### 3️⃣ Selecting the Number of Clusters (K)
**Script:** `3B_Global_K.py`

- Elbow method (distortion & inertia)
- PCA-based visualization of embeddings
- Helps determine the approximate number of identities

---

### 4️⃣ Face Clustering
**Script:** `4B_Clustering.py`

- KMeans clustering on face embeddings
- Each cluster represents a single person
- Clustered faces are organized into folders

---

### 5️⃣ Silhouette Analysis
**Script:** `5B_SilhouetteScore.py`

- Computes silhouette scores per sample and per cluster
- Evaluates cluster quality
- Generates visual diagnostic plots

---

### 6️⃣ Supervised Classification
**Script:** `6B_TrainClassifier.py`

The following classifiers are trained:
- **Centroid-based classifier** (cosine similarity + threshold)
- **k-Nearest Neighbors**
- **Support Vector Machine (linear)**
- **Logistic Regression**

All models are saved to the `./models` directory.

---

### 7️⃣ Submission Generation
**Script:** `7B_GenerateSubmission.py`

- Detects faces in test images
- Extracts embeddings
- Predicts identities using all trained models
- Sorts faces from left to right
- Generates submission CSV files per model

---

## 📦 Install dependencies

### 1. Clone the repository
```bash
pip install -r requirements.txt
```

## 📦 Install dependencies

- python 1B_DetectFaces.py
- python 2B_CreateEmbeddings.py
- python 3B_Global_K.py
- python 4B_Clustering.py
- python 5B_SilhouetteScore.py
- python 6B_TrainClassifier.py
- python 7B_GenerateSubmission.py




