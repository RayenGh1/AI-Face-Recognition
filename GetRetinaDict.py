from uniface import RetinaFace
from uniface.constants import RetinaFaceWeights
import cv2
from pprint import pprint

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "./trainset/0001.jpg"  # pas aan naar 1 foto

# -----------------------------
# LOAD MODEL
# -----------------------------
detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34)

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError(f"Kan afbeelding niet laden: {IMAGE_PATH}")

# -----------------------------
# DETECT FACES
# -----------------------------
faces = detector.detect(image)

print(f"\nAantal gezichten gevonden: {len(faces)}\n")

# -----------------------------
# PRINT RESULT
# -----------------------------
for i, face_data in enumerate(faces):
    print(f"========== FACE {i} ==========")
    pprint(face_data)
    print()
