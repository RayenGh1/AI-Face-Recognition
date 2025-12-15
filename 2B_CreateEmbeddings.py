import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from uniface import ArcFace
from uniface.constants import ArcFaceWeights


# ---------------- CONFIG ----------------
CROPPED_FACES = "./cropped_faces"
OUT_FILE = "faceEmbeddings.parquet"

# Pretrained ArcFace embedder
embedder = ArcFace(model_name=ArcFaceWeights.RESNET)


# ---------------- STORAGE ----------------
embeddings = [] # Stores numerical embedding vectors
filenames = [] # Stores corresponding filenames of the face images


# ---------------- PROCESS EACH FACE IMAGE ----------------
for fname in tqdm(os.listdir(CROPPED_FACES), desc="Embedding faces"):

    # Only process image files
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(CROPPED_FACES, fname) # Path of the image
    img = cv2.imread(path)

    # Skip unreadable files
    if img is None:
        print(f"[SKIP] Can't read {fname}")
        continue

    # ArcFace expects aligned 112x112 faces
    if img.shape[0] != 112 or img.shape[1] != 112:
        print(f"[SKIP] Wrong shape: {img.shape}")
        continue

    # Debug info
    print(f"\nFile: {fname}")
    print(f"shape: {img.shape}")
    print(f"dtype: {img.dtype}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Generate embedding
    emb = embedder.get_embedding(img_rgb).squeeze()

    # Debug info
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding sample: {emb[:5]}")

    embeddings.append(emb)
    filenames.append(fname)


# ---------------- BUILD DATAFRAME ----------------
df = pd.DataFrame(embeddings)
df.columns = [f"dim_{i}" for i in range(df.shape[1])] # Name embedding dimensions (dim_0, dim_1, ...)
df.insert(0, "file", filenames)

print("\nFirst 3 rows:")
print(df.head(3))


# ---------------- SAVE EMBEDDINGS ----------------
df.to_parquet(OUT_FILE, index=False)
print(f"[OK] Saved embeddings: {OUT_FILE} ({len(df)} faces)")
