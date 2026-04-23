"""
╔═════════════════════════════════════════╗
║  MODULE 03 — Build Embedding Database   ║
╚═════════════════════════════════════════╝
Đọc tất cả ảnh trong dataset/ → tạo embedding vector → lưu file .pkl

Dùng: python 03_train.py
      python 03_train.py --rebuild   (xây lại từ đầu)
      python 03_train.py --name nguyen_van_a  (chỉ update 1 người)
"""
import os
import glob
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from deepface import DeepFace
from config import (
    DATASET_DIR, EMBED_FILE, MODEL_NAME, DETECTOR
)


# ── Tạo embedding cho 1 ảnh ───────────────────────────────────────────────

def get_embedding(img_path: str, model: str, detector: str) -> np.ndarray | None:
    """
    Trả về embedding vector 512-dim (ArcFace) hoặc None nếu không detect được.
    enforce_detection=False → không raise lỗi khi không thấy mặt, trả về None.
    """
    try:
        result = DeepFace.represent(
            img_path      = img_path,
            model_name    = model,
            detector_backend = detector,
            enforce_detection = False,
            align         = True,
        )
        if result and result[0].get('face_confidence', 1) > 0.5:
            return np.array(result[0]['embedding'], dtype=np.float32)
        return None
    except Exception:
        return None


# ── Build/Update database ─────────────────────────────────────────────────

def build_database(persons: list[str] | None = None,
                   rebuild: bool = False) -> dict:
    """
    Tạo hoặc cập nhật embedding database.

    Structure của DB:
    {
      'nguyen_van_a': {
          'embeddings': [array, array, ...],   # tất cả embeddings
          'mean':       array,                  # embedding trung bình
          'count':      int
      },
      ...
    }
    """
    # Load DB cũ nếu có và không rebuild
    db = {}
    if EMBED_FILE.exists() and not rebuild:
        with open(EMBED_FILE, 'rb') as f:
            db = pickle.load(f)
        print(f'📂 Đã load DB cũ: {len(db)} người')

    # Xác định danh sách người cần xử lý
    if persons is None:
        persons = [d.name for d in DATASET_DIR.iterdir() if d.is_dir()]

    if not persons:
        print(f'❌ Không tìm thấy dữ liệu trong {DATASET_DIR}/')
        print('   Chạy trước: python 02_extract.py --name <tên>')
        return db

    print(f'\n🧠 Model: {MODEL_NAME}  |  Detector: {DETECTOR}')
    print(f'👥 Sẽ xử lý: {persons}\n')

    for person in persons:
        person_dir = DATASET_DIR / person
        imgs = sorted(glob.glob(str(person_dir / '*.jpg')) +
                      glob.glob(str(person_dir / '*.png')))

        if not imgs:
            print(f'⚠️  {person}: không có ảnh — bỏ qua')
            continue

        print(f'👤 {person}: {len(imgs)} ảnh')
        embeddings = []
        failed = 0

        for img_path in tqdm(imgs, desc=f'   {person}', leave=False):
            emb = get_embedding(img_path, MODEL_NAME, DETECTOR)
            if emb is not None:
                embeddings.append(emb)
            else:
                failed += 1

        if not embeddings:
            print(f'   ❌ Không tạo được embedding nào!')
            continue

        mean_emb = np.mean(embeddings, axis=0)
        # L2-normalize mean embedding
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        db[person] = {
            'embeddings': embeddings,
            'mean':       mean_emb,
            'count':      len(embeddings)
        }
        print(f'   ✅ {len(embeddings)} embeddings OK  |  {failed} ảnh lỗi')

    # Lưu DB
    EMBED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBED_FILE, 'wb') as f:
        pickle.dump(db, f)

    print(f'\n💾 Đã lưu: {EMBED_FILE}')
    _print_summary(db)
    return db


def _print_summary(db: dict):
    from scipy.spatial.distance import cosine

    print(f'\n{"═"*50}')
    print(f'  DATABASE SUMMARY — {MODEL_NAME}')
    print(f'{"═"*50}')
    persons = list(db.keys())
    for p in persons:
        print(f'  {p}: {db[p]["count"]} embeddings')

    if len(persons) < 2:
        print(f'{"═"*50}\n')
        return

    # Ma trận khoảng cách giữa các người
    print(f'\n  Khoảng cách cosine giữa các người:')
    print(f'  (Lý tưởng: > 0.35 để phân biệt tốt)')
    header = '        ' + '  '.join(f'{p[:8]:>8}' for p in persons)
    print(f'  {header}')

    for p1 in persons:
        row = f'  {p1[:8]:<8}'
        for p2 in persons:
            if p1 == p2:
                row += f'  {"—":>8}'
            else:
                dist = cosine(db[p1]['mean'], db[p2]['mean'])
                flag = '✅' if dist > 0.35 else '⚠️'
                row += f'  {dist:.3f}{flag}'
        print(row)

    print(f'{"═"*50}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build embedding database')
    parser.add_argument('--rebuild', action='store_true',
                        help='Xây lại từ đầu (bỏ DB cũ)')
    parser.add_argument('--name', default=None,
                        help='Chỉ update 1 người')
    args = parser.parse_args()

    persons = [args.name] if args.name else None
    db = build_database(persons=persons, rebuild=args.rebuild)

    if db:
        print(f'➡️  Bước tiếp: python 04_recognize.py')
