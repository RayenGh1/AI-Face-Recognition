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

MODE = 0
IMAGEDIR = "./testset"
MODEL_DIR = "./models"
THRESHOLD = 0.10  # for centroid
MODELS_TO_RUN = ["centroid", "knn", "svm", "logistic"]

detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34)
embedder = ArcFace(model_name=ArcFaceWeights.RESNET)

with open(os.path.join(MODEL_DIR, "centroid_labels.pkl"), "rb") as f:
    label_names = pickle.load(f)

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

model_name = "knn"

if MODE == 0:
    cv2.namedWindow("display", cv2.WINDOW_NORMAL)

    for filename in os.listdir(IMAGEDIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(IMAGEDIR, filename)
            img = cv2.imread(path)

            if img is None:
                continue

            # scale image to screen
            h, w = img.shape[:2]
            scale = min(1280 / w, 800 / h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

            faces = detector.detect(img)
            for face in faces:
                x1, y1, x2, y2 = map(int, face["bbox"])
                landmarks = np.array(face["landmarks"], dtype=np.float32)

                aligned, _ = face_alignment(img, landmarks, 112)

                emb = embedder.get_embedding(
                    cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                ).squeeze()
                emb = emb / np.linalg.norm(emb)

                if model_name == "centroid":
                    sims = models["centroid"] @ emb
                    idx = int(np.argmax(sims))
                    score = float(sims[idx])
                    pred = label_names[idx] if score > THRESHOLD else "none"
                else:
                    clf = models[model_name]
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba([emb])[0]
                        pred_id = int(np.argmax(probs))
                        score = float(probs[pred_id])
                        pred = label_names[pred_id]
                    else:
                        pred_id = int(clf.predict([emb])[0])
                        score = 0.0
                        pred = label_names[pred_id]

                label_text = f"{pred} ({score:.2f})"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("display", img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    cv2.destroyAllWindows()



if MODE == 1:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1500, 1000))
        frame = cv2.flip(frame, 1)

        faces = detector.detect(frame)

        for face in faces:
            x1, y1, x2, y2 = map(int, face["bbox"])
            landmarks = np.array(face["landmarks"], dtype=np.float32)

            aligned, _ = face_alignment(frame, landmarks, 112)

            emb = embedder.get_embedding(
                cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            ).squeeze()
            emb = emb / np.linalg.norm(emb)

            if model_name == "centroid":
                sims = models["centroid"] @ emb
                idx = int(np.argmax(sims))
                score = float(sims[idx])
                pred = label_names[idx] if score > THRESHOLD else "none"
            else:
                clf = models[model_name]
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba([emb])[0]
                    pred_id = int(np.argmax(probs))
                    score = float(probs[pred_id])
                    pred = label_names[pred_id]
                else:
                    pred_id = int(clf.predict([emb])[0])
                    score = 0.0
                    pred = label_names[pred_id]

            label_text = f"{pred} ({score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Input", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
