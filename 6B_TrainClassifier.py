import os
import cv2
import numpy as np
import pickle
import json
from tqdm import tqdm
from uniface import ArcFace
from uniface.constants import ArcFaceWeights
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ---------------- CONFIG ----------------
CLUSTER_DIR = "./clusters"
MODEL_DIR = "./models"
JUNK_LABELS = ["unknown"]

os.makedirs(MODEL_DIR, exist_ok=True)

EMBEDDER = ArcFace(model_name=ArcFaceWeights.RESNET)


# ---------------- LOAD TRAINING DATA ----------------
X = [] # embeddings
y = [] # numeric labels
label_names = [] # numeric id -> person name

label_map = {}   # person_name -> numeric id
label_counter = 0

# Iterate over cluster folders
for person in sorted(os.listdir(CLUSTER_DIR)):

    # Skip mixed clusters
    if person in JUNK_LABELS:
        print(f"[SKIP] Skipping junk cluster: {person}")
        continue

    person_path = os.path.join(CLUSTER_DIR, person)
    if not os.path.isdir(person_path):
        continue

    # Assign numeric label
    label_map[person] = label_counter
    label_names.append(person)
    label_counter += 1

    # Load each aligned face in the cluster
    for img_file in tqdm(os.listdir(person_path), desc=f"Loading {person}", leave=False):
        if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # ArcFace expects aligned 112x112 faces
        if img.shape[:2] != (112, 112):
            continue

        # Extract and normalize embedding
        emb = EMBEDDER.get_embedding(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).squeeze()
        emb = emb / (np.linalg.norm(emb) + 1e-12) # +1e-12 to avoid dividing by 0

        X.append(emb)
        y.append(label_map[person])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"[INFO] Training samples: {len(X)}")
print(f"[INFO] Classes: {len(label_names)}")


# ---------------- CENTROID PER PERSON ----------------
centers = [] # List to save cluster centers

for class_id in range(len(label_names)):
    class_embeddings = X[y == class_id] # Take all embeddings X where label y equals class ID

    # Mean embedding per class
    center = np.mean(class_embeddings, axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    centers.append(center)

centers = np.stack(centers)

# Save centroid model
np.save(os.path.join(MODEL_DIR, "centroid_centers.npy"), centers)
with open(os.path.join(MODEL_DIR, "centroid_labels.pkl"), "wb") as f:
    pickle.dump(label_names, f)

print("[OK] Centroid model saved")


# ---------------- MODEL 2 — k-NN CLASSIFIER ----------------
knn = KNeighborsClassifier(n_neighbors=3, metric="cosine") # metric=cosine: use cosine distance to measure

knn.fit(X, y)

# Save knn model
with open(os.path.join(MODEL_DIR, "knn.pkl"), "wb") as f:
    pickle.dump(knn, f)

print("[OK] k-NN model saved")


# ---------------- MODEL 3 — SVM (cosine via linear kernel on normalized vectors) ----------------
svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X, y)

# Save svm model
with open(os.path.join(MODEL_DIR, "svm.pkl"), "wb") as f:
    pickle.dump(svm, f)

print("[OK] SVM model saved")


# ---------------- # MODEL 4 — LOGISTIC REGRESSION ----------------
logreg = LogisticRegression(max_iter=2000, random_state=42)
logreg.fit(X, y)

# Save lr model
with open(os.path.join(MODEL_DIR, "logistic.pkl"), "wb") as f:
    pickle.dump(logreg, f)

print("[OK] Logistic Regression model saved")




print("[DONE] All models trained and saved")