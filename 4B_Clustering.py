import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os
from tqdm import tqdm
import shutil
import pickle


# ---------------- CONFIG ----------------
K = 13 # check 3B_Clustering.py

CROPPED_DIR = "./cropped_faces"
CLUSTER_DIR = "./clusters"
EMBED_FILE = "faceEmbeddings.parquet"


# ---------------- LOAD EMBEDDINGS ----------------
df = pd.read_parquet(EMBED_FILE)
files = df["file"].tolist()
embeddings = df.drop("file", axis=1).to_numpy()

# Debug info
print("Amount embeddings:", embeddings.shape[0])
print("Embedding dimension:", embeddings.shape[1])


# ---------------- NORMALIZE EMBEDDINGS ----------------
# Normalise embeddings -> calculate L2 length of vector and divide with own norm = normalised vector with length 1
# axis = 1 -> for each row
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


# ---------------- KMEANS CLUSTERING ----------------
kmeans = KMeans(n_clusters=K, n_init=20, random_state=42)

# fit -> learn cluster centra from embeddings
# predict -> put every embedding in closest cluster
labels = kmeans.fit_predict(embeddings) # -> labels = 1D array with cluster ID for each embedding

# Debug info
print("\nKMeans info:")
print("Amount clusters:", kmeans.n_clusters)
print("Inertia:", kmeans.inertia_)
print("Amount iterations:", kmeans.n_iter_)


# ---------------- SAVE MODEL OUTPUT ----------------
np.save("labels.npy", labels)

with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)


# ---------------- CREATE CLUSTER FOLDERS ----------------
os.makedirs(CLUSTER_DIR, exist_ok=True)

for file, label in tqdm(zip(files, labels), total=len(files), desc="Assigning clusters"): # Link every file with label (cluster ID) and go over all of them

     # Create one folder per cluster
    dst_dir = os.path.join(CLUSTER_DIR, f"person_{label}")
    os.makedirs(dst_dir, exist_ok=True)

    src = os.path.join(CROPPED_DIR, file) # Path to original face
    dst = os.path.join(dst_dir, file) # Path within goal cluster folder

    # Copy face image into its cluster folder
    if os.path.exists(src):
        shutil.copy(src, dst)

print("[OK] Clusters created")
