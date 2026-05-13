"""
╔══════════════════════════════════════════════════════════════╗
║  MAIN PIPELINE — Tổng hợp quy trình đăng ký khuôn mặt         ║
╚══════════════════════════════════════════════════════════════╝
Luồng chạy:
  1. Hỏi tên ngườii
  2. Ghi video qua camera điện thoại (01_byphone)
  3. Trích xuất ảnh khuôn mặt từ video (02_extract)
  4. Augment ảnh từ predata → dataset (02_augment)
  5. Train/update embedding database (03_train)

Dùng: python main.py
      python main.py --enrich nguyen_van_a
"""
import sys
import os
import argparse

# Thêm thư mục hiện tại vào sys.path để import các module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

import importlib.util

def load_module(module_name: str, file_path: str):
    """Load a Python module dynamically from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ── Đường dẫn tới các module pipeline ─────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Bước 1: Record video ──────────────────────────────────────────────────
_byphone = load_module("_byphone", os.path.join(BASE, "01_byphone.py"))
record_video_by_phone = _byphone.record_video_by_phone

# ── Bước 2: Extract faces ─────────────────────────────────────────────────
_extract = load_module("_extract", os.path.join(BASE, "02_extract.py"))
extract_faces = _extract.extract_faces
find_latest_video = _extract.find_latest_video

# ── Bước 3: Augment images ────────────────────────────────────────────────
_augment = load_module("_augment", os.path.join(BASE, "02_augment.py"))
augment_images = _augment.augment_images

# ── Bước 4: Train / Build database ────────────────────────────────────────
_train = load_module("_train", os.path.join(BASE, "03_train.py"))
build_database = _train.build_database


def run_pipeline():
    """Chạy toàn bộ pipeline đăng ký khuôn mặt (ghi mới)."""

    print("=" * 55)
    print("  🎯 FACE REGISTRATION PIPELINE")
    print("=" * 55)

    # ── 1. Nhập tên ngườii ───────────────────────────────────────────────
    print("\n📋 Bước 1: Nhập thông tin")
    name = input("👤 Nhập tên ngườii (vd: nguyen_van_a): ").strip()
    if not name:
        print("❌ Tên không được để trống!")
        return

    _run_steps(name, mode="new")


def enrich_person(name: str):
    """
    Ghi thêm dữ liệu cho ngườii đã có trong thư viện.
    Extract tự động ghi tiếp từ index cuối, augment chỉ ảnh mới.
    """
    print("=" * 55)
    print(f"  🔄 ENRICH MODE — Thêm dữ liệu cho: {name}")
    print("=" * 55)

    if not name:
        print("❌ Tên không được để trống!")
        return

    _run_steps(name, mode="enrich")


def _run_steps(name: str, mode: str = "new"):
    """Chạy các bước chung cho cả new và enrich."""

    print(f"\n🚀 Bắt đầu pipeline cho: {name} (mode={mode})")
    print("─" * 55)

    # ── 2. Ghi video từ camera điện thoại ────────────────────────────────
    print("\n📱 Bước 2: Ghi video từ camera điện thoại")
    video_path = record_video_by_phone(name)
    if not video_path:
        print("❌ Ghi video thất bại. Dừng pipeline.")
        return

    # ── 3. Trích xuất ảnh khuôn mặt ──────────────────────────────────────
    print("\n🎬 Bước 3: Trích xuất ảnh khuôn mặt")
    n_extracted = extract_faces(name, video_path)
    if n_extracted == 0:
        print("❌ Không trích xuất được ảnh nào. Dừng pipeline.")
        return

    # ── 4. Augment ảnh (predata → dataset) ───────────────────────────────
    print("\n🔄 Bước 4: Augment ảnh")
    # Nếu enrich → chỉ augment ảnh mới (skip_existing=True)
    augment_images(name, skip_existing=(mode == "enrich"))

    # ── 5. Train / Update embedding database ─────────────────────────────
    print("\n🧠 Bước 5: Build/Update embedding database")
    db = build_database(persons=[name], rebuild=False)

    if not db or name not in db:
        print("❌ Train thất bại. Kiểm tra lại dữ liệu.")
        return

    # ── Hoàn tất ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  ✅ HOÀN TẤT — {name}")
    print(f"  📁 Video    : {video_path}")
    print(f"  📷 Extract  : {n_extracted} ảnh → predata/{name}/")
    print(f"  🎨 Augment  : → dataset/{name}/")
    print(f"  🧠 Embeddings: {db[name]['count']} vectors")
    print("=" * 55)
    print("\n➡️  Bước tiếp: python 04_recognize.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Registration Pipeline")
    parser.add_argument("--enrich", type=str, default=None,
                        help="Tên ngườii đã có để ghi thêm dữ liệu (vd: nguyen_van_a)")
    args = parser.parse_args()

    if args.enrich:
        enrich_person(args.enrich)
    else:
        run_pipeline()

