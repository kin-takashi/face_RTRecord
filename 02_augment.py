import os
import cv2
from albumentations import (
    Compose, RandomBrightnessContrast, GaussNoise,
    MotionBlur, Rotate, HorizontalFlip
)
from tqdm import tqdm
from config import (DATAPREP_DIR,DATASET_DIR)


def augment_images(name: str):

 INPUT_DIR = DATAPREP_DIR / name
 OUTPUT_DIR = DATASET_DIR / name

 os.makedirs(OUTPUT_DIR, exist_ok=True)

 transform = Compose([
    RandomBrightnessContrast(p=0.5),  # light
    GaussNoise(p=0.3),                # noise
    MotionBlur(p=0.2),                # blur
    Rotate(limit=20, p=0.5),          # rotate
    HorizontalFlip(p=0.5)             # flip
 ])

 # 🔄 Số ảnh tạo thêm mỗi ảnh gốc
 AUG_PER_IMAGE = 5

 # 🚀 Xử lý
 for img_name in tqdm(os.listdir(INPUT_DIR), desc="Augmenting"):
    img_path = os.path.join(INPUT_DIR, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    name, ext = os.path.splitext(img_name)

    for i in range(AUG_PER_IMAGE):
        augmented = transform(image=img)["image"]

        new_name = f"{name}_aug_{i}{ext}"
        save_path = os.path.join(OUTPUT_DIR, new_name)

        cv2.imwrite(save_path, augmented)