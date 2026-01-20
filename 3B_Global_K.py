import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import os


# ---------------- CONFIG ----------------
EMBED_FILE = "faceEmbeddings.parquet"
PLOTS_DIR = "./k_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
K_range = range(5, 16)


# ---------------- LOAD EMBEDDINGS ----------------
df = pd.read_parquet(EMBED_FILE)
embeddings = df.drop("file", axis=1).to_numpy()

# Normalise embeddings -> calculate L2 length of vector and divide with own norm = normalised vector with length 1
# axis = 1 -> for each row
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


# ---------------- ELBOW ANALYSIS ----------------
distortions = []
inertias = []

for k in tqdm(K_range, desc="Testing K"):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)

    # Distortion (mean squared distance to closest center)

    # cdist: calculate euclidic distance from each point to cluster center
    # np.min: take the closest cluster center forn each point
    # **2 square the distance -> so outliers have a larger punishment
    # npmean: take the avarage
    distortion = np.mean(
        np.min(cdist(embeddings, kmeans.cluster_centers_, "euclidean"), axis=1) ** 2
    )
    distortions.append(distortion)

    # Inertia (sum of squared distances)
    inertias.append(kmeans.inertia_)


# ---------------- PLOT ELBOW (DISTORTION) ----------------
plt.figure()
plt.plot(list(K_range), distortions, marker="x")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Distortion")
plt.title("Elbow Method (Distortion)")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "elbow_distortion.png"), dpi=200)
plt.show()


# ---------------- PLOT ELBOW (INERTIA) ----------------
plt.figure()
plt.plot(list(K_range), inertias, marker="x")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method (Inertia)")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "elbow_inertia.png"), dpi=200)
plt.show()


# ---------------- PCA PROJECTION ----------------
pca = PCA(n_components=2, random_state=42, n_init=20) # components=2 -> 2D
embeddings_2d = pca.fit_transform(embeddings)


# ---------------- PCA VISUALISATION (NO CLUSTERS) ----------------
plt.figure()
plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c="gray",
    s=12,
    alpha=0.7
)
plt.title("PCA of face embeddings (no clustering)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pca_no_clustering.png"), dpi=200)


# ---------------- PCA VISUALISATION (NO CLUSTERS) ----------------
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    plt.figure()
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="viridis", s=12)
    plt.title(f"KMeans clustering in PCA space (k={k})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"pca_k_{k}.png"), dpi=200)

print(f"[OK] Elbow + PCA analysis completed: {PLOTS_DIR}")
