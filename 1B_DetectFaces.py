from uniface import RetinaFace
from uniface.face_utils import face_alignment
from uniface.constants import RetinaFaceWeights
import cv2
import os
import numpy as np
from tqdm import tqdm
import csv

# ---------------- CONFIG ----------------
CONF_THRESHOLD = 0.7  # minimum confidence

# Face detector (pretrained)
detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34)

# Paths
TRAIN_PATH = "./trainset"
CROPPED_FACES = "./cropped_faces"
os.makedirs(CROPPED_FACES, exist_ok=True)
META_CSV = os.path.join(CROPPED_FACES, "faces_metadata.csv")

# Visual window
cv2.namedWindow("display", cv2.WINDOW_NORMAL)


# ---------------- COLLECT IMAGES ----------------
image_filenames = [
    f for f in os.listdir(TRAIN_PATH)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]


# ---------------- OPEN METADATA CSV ----------------
with open(META_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "aligned_file",
        "original_file",
        "face_index",
        "x1", "y1", "x2", "y2",
        "confidence"
    ])


# ---------------- PROCESS EACH IMAGE ----------------
    for image_index, image_filename in enumerate(tqdm(image_filenames, desc="Processing images")):
        full_path = os.path.join(TRAIN_PATH, image_filename)
        image = cv2.imread(full_path)

        # Skip unreadable images
        if image is None:
            print(f"[WARN] Error reading {full_path}")
            continue
        
        # Detect faces in the image -> result is list of detected faces data
        faces = detector.detect(image)

        # copy the image to draw
        display_image = image.copy()

        # No faces detected -> just show image
        if len(faces) == 0:
            cv2.imshow("display", display_image)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                raise SystemExit
            continue


# ---------------- PROCESS EACH FACE ----------------  
        for face_index, face_data in enumerate(faces):

            # Read confidence score
            score = (face_data.get("confidence"))

            # Bounding box coords
            x1, y1, x2, y2 = map(int, face_data["bbox"])
            w, h = (x2 - x1), (y2 - y1)


# ---------------- VISUALIZATION ----------------
            # draw bounding box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label_text = f"{score:.2f}"

            # Draw confidence score
            cv2.putText(
                display_image,
                label_text,
                (x1, max(0, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (255, 255, 255),
                6,
                cv2.LINE_AA
            )

            # Draw facial landmarks
            for lm in face_data["landmarks"]:
                cv2.circle(
                    display_image,
                    (int(lm[0]), int(lm[1])),
                    3,
                    (255, 0, 0),
                    -1
                )

# ---------------- LOGIC ----------------
            # Skip if confidence is too low
            if score < CONF_THRESHOLD: 
                continue # Next image

            # Skip small faces
            if w < 40 or h < 40:
                continue # Next image

            landmarks = np.array(face_data["landmarks"], dtype=np.float32)

            # Align face to 112x112 using landmarks
            aligned_face, _ = face_alignment(image, landmarks, 112)

            aligned_name = f"aligned_{image_index}_{face_index}.png"
            save_path = os.path.join(CROPPED_FACES, aligned_name)
            cv2.imwrite(save_path, aligned_face)

            # Save metadata
            writer.writerow([
                aligned_name,
                image_filename,
                face_index,
                x1, y1, x2, y2,
                score
            ])

# ---------------- UPDATE DISPLAY ----------------
        cv2.imshow("display", display_image)
        key = cv2.waitKey(1)

        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit

cv2.destroyAllWindows()
print(f"[OK] Saved faces + metadata in {META_CSV}")
