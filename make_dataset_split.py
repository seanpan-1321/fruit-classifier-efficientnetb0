from pathlib import Path
import shutil
import random
from tqdm import tqdm

# =====================
# Paths
# =====================
ORIGINAL_DIR = Path(r"D:\food_dataset_resized")
FRUITS262_DIR = Path(r"D:\fruit_project_fruit-262\208x256")

RAW_OUT = Path(r"D:\fruit_project\clean_fruit_raw")
SPLIT_OUT = Path(r"D:\fruit_project\clean_fruit_split")

# =====================
# Settings
# =====================
EXTRA_IMAGES_PER_WEAK_CLASS = 500

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

WEAK_CLASS_MAP = {
    "Apple": "apple",
    "Apricot": "apricot",
    "Mango": "mango",
    "Banana": "banana",
    "Orange": "orange",
    "Dates": "date",
    "Guava": "guava",
    "Pear": "pear",
    "Watermelon": "watermelon",
    "Zucchini": "zucchini",
}

# =====================
# Safety
# =====================
if RAW_OUT.exists():
    raise FileExistsError(f"Delete this folder first: {RAW_OUT}")

if SPLIT_OUT.exists():
    raise FileExistsError(f"Delete this folder first: {SPLIT_OUT}")

RAW_OUT.mkdir(parents=True, exist_ok=True)

# =====================
# 1. Combine original train + test
# =====================
print("\nCombining original train + test...")

for split_name in ["train", "test"]:
    split_dir = ORIGINAL_DIR / split_name

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing folder: {split_dir}")

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        target_dir = RAW_OUT / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        image_files = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        ]

        for i, img_path in enumerate(tqdm(image_files, desc=f"{split_name}/{class_name}")):
            new_name = f"orig_{split_name}_{class_name}_{i}{img_path.suffix.lower()}"
            shutil.copy2(img_path, target_dir / new_name)

# =====================
# 2. Add Fruits-262 images to weak classes
# =====================
print("\nAdding Fruits-262 images to weak classes...")

for old_class, fruits262_class in WEAK_CLASS_MAP.items():
    source_dir = FRUITS262_DIR / fruits262_class
    target_dir = RAW_OUT / old_class

    if not source_dir.exists():
        print(f"Skipping {old_class}: Fruits-262 source not found: {source_dir}")
        continue

    if not target_dir.exists():
        print(f"Skipping {old_class}: original class not found: {target_dir}")
        continue

    source_images = [
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]

    random.shuffle(source_images)
    selected = source_images[:EXTRA_IMAGES_PER_WEAK_CLASS]

    copied = 0

    for i, img_path in enumerate(tqdm(selected, desc=f"Fruits-262/{old_class}")):
        new_name = f"f262_{old_class}_{i}{img_path.suffix.lower()}"
        shutil.copy2(img_path, target_dir / new_name)
        copied += 1

    print(f"{old_class}: added {copied} Fruits-262 images")

# =====================
# 3. Print raw counts before splitting
# =====================
print("\nRaw dataset counts before split:")
raw_counts = {}

for class_dir in sorted(RAW_OUT.iterdir()):
    if not class_dir.is_dir():
        continue

    count = len([
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ])

    raw_counts[class_dir.name] = count
    print(f"{class_dir.name}: {count}")

# =====================
# 4. Split into train / val / test
# =====================
print("\nCreating new train / val / test split...")

for class_dir in sorted(RAW_OUT.iterdir()):
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_files = images[:train_count]
    val_files = images[train_count:train_count + val_count]
    test_files = images[train_count + val_count:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split, files in split_map.items():
        target_dir = SPLIT_OUT / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(files):
            new_name = f"{split}_{class_name}_{i}{img_path.suffix.lower()}"
            shutil.copy2(img_path, target_dir / new_name)

# =====================
# 5. Print final split counts
# =====================
print("\nFinal split counts:")
for split in ["train", "val", "test"]:
    split_dir = SPLIT_OUT / split
    total_split = 0

    print(f"\n{split.upper()}:")

    for class_dir in sorted(split_dir.iterdir()):
        count = len([
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        ])
        total_split += count
        print(f"{class_dir.name}: {count}")

    print(f"{split} total: {total_split}")

print("\nDone.")
print(f"Raw dataset: {RAW_OUT}")
print(f"New split dataset: {SPLIT_OUT}")
print(f'Use this in training: DATA_DIR = Path(r"{SPLIT_OUT}")')