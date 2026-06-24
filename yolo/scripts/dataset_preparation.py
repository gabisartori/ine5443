from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
from PIL import Image
import numpy as np

# ----------------------------
# Configuration
# ----------------------------

CSV_FILE = "GroundTruth.csv"
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("yolo/dataset")
MASKS_DIR = Path("masks")

CLASS_COLUMNS = [
  "MEL",
  "NV",
  "BCC",
  "AKIEC",
  "BKL",
  "DF",
  "VASC",
]

RANDOM_STATE = 42

# ----------------------------
# Read CSV
# ----------------------------

df = pd.read_csv(CSV_FILE)

# Determine class from one-hot encoding
df["class"] = df[CLASS_COLUMNS].idxmax(axis=1)

# ----------------------------
# Stratified split
# ----------------------------

train_df, val_df = train_test_split(
  df,
  test_size=0.20,
  random_state=RANDOM_STATE,
  stratify=df["class"],
)

# ----------------------------
# Create directory structure
# ----------------------------

for split in ["train", "val"]:
  for cls in CLASS_COLUMNS:
    (OUTPUT_DIR / "images" / split / cls).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" /  split / cls).mkdir(parents=True, exist_ok=True)

# ----------------------------
# Get image bounding box based on mask file
# ----------------------------

def get_bounding_box(mask_path):

  mask = Image.open(mask_path)
  mask_array = np.array(mask)

  # Get the coordinates of the non-zero pixels
  coords = np.column_stack(np.where(mask_array > 0))

  if coords.size == 0:
    return None

  # Get the bounding box coordinates
  y_min, x_min = coords.min(axis=0)
  y_max, x_max = coords.max(axis=0)

  # Normalize the bounding box coordinates to [0, 1] range
  height, width = mask_array.shape
  x_min /= width
  x_max /= width
  y_min /= height
  y_max /= height

  # Return the bounding box coordinates in YOLO format (x_center, y_center, width, height)
  x_center = (x_min + x_max) / 2
  y_center = (y_min + y_max) / 2
  width = x_max - x_min
  height = y_max - y_min

  return x_center, y_center, width, height


# ----------------------------
# Create symlinks
# ----------------------------

def populate_split(split_df, split_name):
  for _, row in split_df.iterrows():
    image_id = row["image"]
    cls = row["class"]

    source = IMAGES_DIR / f"{image_id}.jpg"
    mask = MASKS_DIR / f"{image_id}_segmentation.png"

    if not source.exists():
      source = IMAGES_DIR / f"{image_id}.png"

    if not source.exists():
      print(f"Missing image: {image_id}")
      continue

    if not mask.exists():
      print(f"Missing mask: {image_id}")
      continue

    target = OUTPUT_DIR / "images" / split_name / cls / source.name

    # Create label file
    bbox = get_bounding_box(mask)
    if bbox is None:
      print(f"No bounding box found for mask: {mask}")
      continue

    label_file = OUTPUT_DIR / "labels" / split_name / cls / f"{image_id}.txt"
    with open(label_file, "w") as f:
      class_index = CLASS_COLUMNS.index(cls)
      f.write(f"{class_index} {' '.join(map(str, bbox))}\n")

    if target.exists() or target.is_symlink():
      continue

    os.symlink(
      source.resolve(),
      target,
    )

populate_split(train_df, "train")
populate_split(val_df, "val")

# ----------------------------
# Generate dataset.yaml
# ----------------------------

yaml_data = {
  "path": str(OUTPUT_DIR.resolve()),
  "train": "train",
  "val": "val",
  "names": CLASS_COLUMNS,
}

with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
  yaml.safe_dump(yaml_data, f, sort_keys=False)

print("Dataset created successfully.")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")