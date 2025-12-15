import os
import cv2
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
from uniface import RetinaFace, ArcFace
from uniface.face_utils import face_alignment
from uniface.constants import RetinaFaceWeights, ArcFaceWeights
from pillow_heif import register_heif_opener
from PIL import Image


# ---------------- CONFIG ----------------
TEST_PATH = "./testset"
MODEL_DIR = "./models"
SUBMISSION_DIR = "./submission"

THRESHOLD = 0.10   # for centroid

MODELS_TO_RUN = ["centroid", "knn", "svm", "logistic"]

register_heif_opener()
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# ---------------- CONVERT HEIC IMAGES TO JPG ----------------
for fname in os.listdir(TEST_PATH):
    if fname.lower().endswith(".heic"):
        path = os.path.join(TEST_PATH, fname)
        jpg = path.rsplit(".", 1)[0] + ".jpg"
        img = Image.open(path).convert("RGB")
        img.save(jpg, "JPEG", quality=95)
        os.remove(path)
        print(f"[CONVERT] {fname} to {os.path.basename(jpg)}")


# ---------------- LOAD FACE MODELS ----------------
detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34)
embedder = ArcFace(model_name=ArcFaceWeights.RESNET)


# ---------------- LOAD LABEL NAMES ----------------
with open(os.path.join(MODEL_DIR, "centroid_labels.pkl"), "rb") as f:
    label_names = pickle.load(f)


# ---------------- LOAD CLASSIFIERS ----------------
models = {}

# Centroid
centers = np.load(os.path.join(MODEL_DIR, "centroid_centers.npy"))
centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
models["centroid"] = centers

# kNN
with open(os.path.join(MODEL_DIR, "knn.pkl"), "rb") as f:
    models["knn"] = pickle.load(f)

# SVM
with open(os.path.join(MODEL_DIR, "svm.pkl"), "rb") as f:
    models["svm"] = pickle.load(f)

# Logistic
with open(os.path.join(MODEL_DIR, "logistic.pkl"), "rb") as f:
    models["logistic"] = pickle.load(f)


# ---------------- LIST TEST IMAGES ----------------
test_images = sorted([
    f for f in os.listdir(TEST_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"[INFO] Found {len(test_images)} test images")


# ---------------- RUN PER MODEL ----------------
for model_name in MODELS_TO_RUN:

    print(f"\n==============================")
    print(f" Running model: {model_name}")
    print(f"==============================")

    predictions = defaultdict(list)

# ---------------- PROCESS EACH IMAGE ----------------
    for img_name in tqdm(test_images, desc=f"Predicting ({model_name})"):
        img = cv2.imread(os.path.join(TEST_PATH, img_name))

        # if image is unreadable -> dummy prediction
        if img is None:
            predictions[img_name].append((99999, "none"))
            continue

        # if no faces -> dummy prediction
        faces = detector.detect(img)
        if len(faces) == 0:
            predictions[img_name].append((99999, "none"))
            continue


# ---------------- PROCESS EACH FACE ----------------
        for face in faces:
            x1, y1, x2, y2 = map(int, face["bbox"]) # Take bbox coords
            landmarks = np.array(face["landmarks"], dtype=np.float32) # Take landmark coords

            # Align face to 112x112
            aligned, _ = face_alignment(img, landmarks, 112)

            # Extract normalized embedding
            emb = embedder.get_embedding(
                cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            ).squeeze()
            emb = emb / np.linalg.norm(emb)

            mid_x = (x1 + x2) / 2.0 # Calculate middle of face to later sort faces left to right


# ---------------- PREDICT ----------------
            if model_name == "centroid":
                sims = models["centroid"] @ emb # Take each centroid embedding (mean of all embeddings of this person) and dot product with embedding of new face
                idx = int(np.argmax(sims)) # Index of person with highest similarity
                score = sims[idx] # How good the match is

                # rejection threshold
                pred = label_names[idx] if score > THRESHOLD else "none"

            else:
                clf = models[model_name] # Take model
                pred_id = clf.predict([emb])[0] # Predict cluster ID
                pred = label_names[pred_id] # Cluster ID -> person name

            predictions[img_name].append((mid_x, pred.lower().strip()))


# ---------------- BUILD CSV FILE ----------------
    rows = ["image,label_name"]

    for img in test_images:
        # Sort faces left to right
        items = sorted(predictions[img], key=lambda x: x[0])

        # Remove duplicates while preserving order
        labels = [lbl for (_, lbl) in items if lbl != "unknown"]

        final = ";".join(dict.fromkeys(labels)) if labels else "none"

        img_id = str(int(os.path.splitext(img)[0]))
        rows.append(f"{img_id},{final}")

    out_file = os.path.join(SUBMISSION_DIR, f"submission_{model_name}.csv")
    with open(out_file, "w") as f:
        f.write("\n".join(rows))

    print(f"[OK] Saved {out_file}")

print("\n[DONE] All submissions created in ./submission/")