# """
# ╔══════════════════════════════════════════╗
# ║  MODULE 02 — Trích xuất khuôn mặt        ║
# ╚══════════════════════════════════════════╝
# Đọc video → phát hiện mặt → lọc chất lượng → lưu ảnh

# Dùng: python 02_extract.py --name nguyen_van_a --video videos/nguyen_van_a_xxx.mp4
#       python 02_extract.py --name nguyen_van_a   (tự tìm video mới nhất)
# """
import cv2
import os
import glob
import argparse
import hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    DATAPREP_DIR, VIDEO_DIR,
    CAPTURE_EVERY_N_FRAMES, MAX_IMAGES_PER_PERSON, MIN_IMAGES_PER_PERSON,
    MIN_FACE_SIZE, BLUR_THRESHOLD, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
    DUPLICATE_THRESHOLD, DETECTOR
)


# ── Bộ phát hiện khuôn mặt ─────────────────────────────────────────────────

def load_detector(backend: str):
    """
    Trả về detector theo backend được chọn trong config.
    'opencv'     → Haar Cascade (nhanh, không cần cài thêm)
    'retinaface' → RetinaFace   (chính xác hơn, cần: pip install retina-face)
    """
    if backend == 'retinaface':
        try:
            from retinaface import RetinaFace as RF
            print('✅ Dùng RetinaFace')
            return ('retinaface', RF)
        except ImportError:
            print('⚠️  Không tìm thấy retina-face → fallback sang OpenCV Haar')

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    print('✅ Dùng OpenCV Haar Cascade')
    return ('opencv', cascade)


def detect_faces(frame, detector_pack):
    """Phát hiện khuôn mặt. Trả về list of (x, y, w, h)."""
    kind, det = detector_pack

    if kind == 'retinaface':
        try:
            faces = det.detect_faces(frame)
            result = []
            for key, val in faces.items():
                x1, y1, x2, y2 = val['facial_area']
                result.append((x1, y1, x2 - x1, y2 - y1))
            return result
        except Exception:
            return []

    # OpenCV Haar
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    rects = det.detectMultiScale(gray, scaleFactor=1.1,
                                  minNeighbors=5, minSize=(60, 60))
    return [(x, y, w, h) for (x, y, w, h) in rects] if len(rects) > 0 else []


# ── Kiểm tra chất lượng ────────────────────────────────────────────────────

def laplacian_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_ok(img, lo, hi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = gray.mean()
    return lo <= m <= hi, m


def phash(img) -> np.ndarray:
    small = cv2.resize(img, (32, 32))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float)
    dct   = cv2.dct(gray)[:8, :8]
    return (dct > np.median(dct)).flatten()


def hamming(h1, h2):
    return int(np.count_nonzero(h1 != h2))


def is_too_similar(face_img, existing_hashes, threshold):
    h = phash(face_img)
    for prev in existing_hashes[-40:]:
        if hamming(h, prev) <= threshold:
            return True
    return False


# ── Pipeline chính ─────────────────────────────────────────────────────────

def extract_faces(name: str, video_path: str,
                  max_images: int = MAX_IMAGES_PER_PERSON) -> int:
    """
    Đọc video, trích xuất khuôn mặt đạt chất lượng, lưu vào dataset/{name}/.
    Trả về số ảnh đã lưu.
    """
    save_dir = DATAPREP_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Đếm ảnh đã có (nếu chạy lại để bổ sung)
    existing = list(save_dir.glob('*.jpg'))
    start_idx = len(existing)
    if start_idx > 0:
        print(f'ℹ️  Đã có {start_idx} ảnh — sẽ bổ sung thêm')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'❌ Không mở được video: {video_path}')
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f'🎬 Video: {total_frames} frames | {fps:.0f}fps | {total_frames/fps:.1f}s')
    print(f'📁 Lưu vào: {save_dir}')

    detector    = load_detector(DETECTOR)
    saved       = 0
    skip_blur   = skip_light = skip_dup = skip_small = skip_noface = 0
    phashes     = []
    frame_idx   = 0

    pbar = tqdm(total=max_images, desc=f'  {name}', unit='ảnh')

    try:
        while saved < max_images:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % CAPTURE_EVERY_N_FRAMES != 0:
                continue

            faces = detect_faces(frame, detector)
            if not faces:
                skip_noface += 1
                continue

            # Lấy mặt lớn nhất
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                skip_small += 1
                continue

            # Thêm padding 15%
            pad = int(min(w, h) * 0.15)
            x1  = max(0, x - pad);  y1 = max(0, y - pad)
            x2  = min(frame.shape[1], x + w + pad)
            y2  = min(frame.shape[0], y + h + pad)
            face = cv2.resize(frame[y1:y2, x1:x2], (224, 224))

            # Kiểm tra chất lượng
            blur_score = laplacian_score(face)
            if blur_score < BLUR_THRESHOLD:
                skip_blur += 1
                continue

            bright_ok, _ = brightness_ok(face, MIN_BRIGHTNESS, MAX_BRIGHTNESS)
            if not bright_ok:
                skip_light += 1
                continue

            if phashes and is_too_similar(face, phashes, DUPLICATE_THRESHOLD):
                skip_dup += 1
                continue

            # Lưu ảnh
            idx  = start_idx + saved
            path = save_dir / f'{name}_{idx:04d}.jpg'
            cv2.imwrite(str(path), face)
            phashes.append(phash(face))
            saved += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()

    # Báo cáo
    print(f'\n{"─"*45}')
    print(f'  ✅ Đã lưu       : {saved} ảnh')
    print(f'  🚫 Không có mặt : {skip_noface} frame')
    print(f'  🚫 Mờ           : {skip_blur}')
    print(f'  🚫 Ánh sáng     : {skip_light}')
    print(f'  🚫 Trùng lặp    : {skip_dup}')
    print(f'  🚫 Quá nhỏ      : {skip_small}')

    total_saved = start_idx + saved
    if total_saved < MIN_IMAGES_PER_PERSON:
        print(f'\n  ⚠️  Chỉ có {total_saved} ảnh (cần tối thiểu {MIN_IMAGES_PER_PERSON})')
        print('     → Quay thêm video với góc / ánh sáng khác')
    else:
        print(f'\n  🎉 Tổng {total_saved} ảnh — đủ để train!')
    print(f'{"─"*45}')

    return saved


def find_latest_video(name: str) -> str | None:
    """Tìm video mới nhất của person trong thư mục videos/."""
    pattern = str(VIDEO_DIR / f'{name}_*.mp4')
    files   = sorted(glob.glob(pattern))
    return files[-1] if files else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trích xuất khuôn mặt từ video')
    parser.add_argument('--name',  required=True)
    parser.add_argument('--video', default=None, help='Đường dẫn video (bỏ trống = tự tìm)')
    parser.add_argument('--max',   type=int, default=MAX_IMAGES_PER_PERSON)
    args = parser.parse_args()

    video_path = args.video or find_latest_video(args.name)
    if not video_path:
        print(f'❌ Không tìm thấy video cho "{args.name}" trong {VIDEO_DIR}/')
        print(f'   Chạy trước: python 01_record.py --name {args.name}')
        exit(1)

    print(f'📹 Video: {video_path}')
    n = extract_faces(args.name, video_path, args.max)
    if n > 0:
        print(f'\n➡️  Bước tiếp: python 03_train.py')
