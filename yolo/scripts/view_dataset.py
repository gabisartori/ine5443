from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------

DATASET_DIR = Path("dataset")

IMAGE_DIR = DATASET_DIR / "images" / "train" / "BCC"
LABEL_DIR = DATASET_DIR / "labels" / "train" / "BCC"

CLASS_NAMES = [
  "MEL",
  "NV",
  "BCC",
  "AKIEC",
  "BKL",
  "DF",
  "VASC",
]

# ----------------------------
# Select random image
# ----------------------------

image_path = random.choice(list(IMAGE_DIR.glob("*.*")))
label_path = LABEL_DIR / f"{image_path.stem}.txt"

image_path = DATASET_DIR / "images" / "train" / "NV" / "ISIC_0024306.jpg"
label_path = DATASET_DIR / "labels" / "train" / "NV" / "ISIC_0024306.txt"

print(f"Image: {image_path}")
print(f"Labels: {label_path}")

# ----------------------------
# Load image
# ----------------------------

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w = image.shape[:2]

# ----------------------------
# Draw boxes
# ----------------------------

if label_path.exists():
  with open(label_path) as f:
    for line in f:
      parts = line.strip().split()

      if len(parts) != 5:
        continue

      cls_id, x_c, y_c, bw, bh = map(float, parts)

      cls_id = int(cls_id)

      # YOLO normalized coords -> pixel coords
      x_c *= w
      y_c *= h
      bw *= w
      bh *= h

      x1 = int(x_c - bw / 2)
      y1 = int(y_c - bh / 2)
      x2 = int(x_c + bw / 2)
      y2 = int(y_c + bh / 2)

      cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        1,
      )

      label = (
        CLASS_NAMES[cls_id]
        if cls_id < len(CLASS_NAMES)
        else str(cls_id)
      )

      cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
      )

# ----------------------------
# Show image
# ----------------------------

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.tight_layout()
plt.show()