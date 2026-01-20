import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------- CONFIG ----------------
EMBEDDINGS_FILE = "faceEmbeddings.parquet"
OUTPUT_FILE = "faceEmbeddings_with_labels.parquet"
PLOTS_DIR = "./silhouette_plots"
LABELS = "./labels.npy"
os.makedirs(PLOTS_DIR, exist_ok=True)

label_map = {}


# ---------------- LOAD EMBEDDINGS ----------------
df = pd.read_parquet(EMBEDDINGS_FILE)

# Ensure clean label column
if "label" in df.columns:
    df = df.drop(columns=["label"])
df["label"] = None


labels = np.load(LABELS)
df["label"] = labels

# ---------------- SAVE LABELED EMBEDDINGS ----------------
df.to_parquet(OUTPUT_FILE, index=False)
print(f"[OK] Updated parquet saved in {OUTPUT_FILE}")


# ---------------- PREPARE DATA FOR SILHOUETTE ----------------
embeddings = df.drop(["file", "label"], axis=1).to_numpy()

# Normalise embeddings -> calculate L2 length of vector and divide with own norm = normalised vector with length 1
# axis = 1 -> for each row
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

labels = df["label"].to_numpy()


# ---------------- SAFETY CHECKS ----------------
unique_labels = set(labels) # Check if theres only unique values, deletes doubles

# Every face must belong to a cluster
if None in unique_labels:
    print("[ERROR] Some faces were not assigned to a cluster.")
    exit()

# Silhouette requires at least two clusters
if len(unique_labels) < 2:
    print("[ERROR] Not enough clusters for silhouette scoring.")
    exit()


# ---------------- CALCULATE SILHOUETTE SCORES ----------------
scores = silhouette_samples(embeddings, labels) # Calculate how well each datapoint fits in their own cluster


# ---------------- SILHOUETTE SCORE PER CLUSTER ----------------
cluster_names = []
cluster_means = []
cluster_sizes = []

print("\n📊 Silhouette score per cluster:")
print("=================================")

for label in unique_labels:
    idx = np.where(labels == label)[0]
    avg_score = scores[idx].mean()
    cluster_names.append(label)
    cluster_means.append(avg_score)
    cluster_sizes.append(len(idx))
    print(f"{label:12d} -> avg={avg_score:.3f} (n={len(idx)})")

plt.figure(figsize=(8, 4))
plt.bar(cluster_names, cluster_means)
plt.axhline(0, color="red", linestyle="--")
plt.ylabel("Average silhouette score")
plt.title("Silhouette score per cluster")
plt.xticks(rotation=45)
plt.tight_layout()
#plt.savefig(os.path.join(PLOTS_DIR, "silhouette_per_cluster.png"), dpi=200)
plt.show()


# ---------------- SILHOUETTE SCORE PER SAMPLE ----------------
plt.figure(figsize=(6, 4))
plt.plot(np.sort(scores))
plt.xlabel("Samples (sorted)")
plt.ylabel("Silhouette score")
plt.title("Silhouette scores per face")
plt.grid(True)
plt.tight_layout()
#plt.savefig(os.path.join(PLOTS_DIR, "silhouette_per_sample.png"), dpi=200)
plt.show()


# ---------------- CLUSTER SIZE DISTRIBUTIONR ----------------
plt.figure(figsize=(8, 4))
plt.bar(cluster_names, cluster_sizes)
plt.ylabel("Number of faces")
plt.title("Cluster sizes")
plt.xticks(rotation=45)
plt.tight_layout()
#plt.savefig(os.path.join(PLOTS_DIR, "cluster_sizes.png"), dpi=200)
plt.show()

print(f"\n[OK] Silhouette analysis + plots saved: {PLOTS_DIR}")
print("[INFO] Done!")