import os
import cv2
from albumentations import (
    Compose, RandomBrightnessContrast, GaussNoise,
    MotionBlur, Rotate, HorizontalFlip
)
from tqdm import tqdm
from config import (DATAPREP_DIR, DATASET_DIR)


def augment_images(name: str, skip_existing: bool = True):
    """
    Augment ảnh từ predata/{name}/ → dataset/{name}/
    
    Args:
        name: Tên ngườii
        skip_existing: Nếu True, chỉ augment ảnh chưa có trong dataset (chống lặp)
    """
    INPUT_DIR = DATAPREP_DIR / name
    OUTPUT_DIR = DATASET_DIR / name

    if not INPUT_DIR.exists():
        print(f"❌ Không tìm thấy thư mục predata/{name}/")
        return

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

    # 📋 Lấy danh sách ảnh đã augment (để check trùng)
    existing_augmented = set()
    if skip_existing and OUTPUT_DIR.exists():
        for f in os.listdir(OUTPUT_DIR):
            existing_augmented.add(f)

    # 🚀 Xử lý — chỉ augment ảnh chưa được augment
    skipped = 0
    processed = 0

    for img_name in tqdm(os.listdir(INPUT_DIR), desc="Augmenting"):
        img_path = os.path.join(INPUT_DIR, img_name)

        # Bỏ qua nếu không phải ảnh
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        name_base, ext = os.path.splitext(img_name)

        # 🔍 Kiểm tra ảnh này đã được augment chưa (check file _aug_0)
        if skip_existing:
            aug0_name = f"{name_base}_aug_0{ext}"
            if aug0_name in existing_augmented:
                skipped += 1
                continue

        # 🎨 Augment và lưu
        for i in range(AUG_PER_IMAGE):
            augmented = transform(image=img)["image"]
            new_name = f"{name_base}_aug_{i}{ext}"
            save_path = os.path.join(OUTPUT_DIR, new_name)
            cv2.imwrite(save_path, augmented)

        processed += 1

    print(f"✅ Đã augment: {processed} ảnh mới | ⏭️  Bỏ qua: {skipped} ảnh cũ")
    print(f"📁 Lưu tại: {OUTPUT_DIR}/")
