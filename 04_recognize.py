# """
# ╔══════════════════════════════════════════╗
# ║  MODULE 04 — Nhận diện khuôn mặt         ║
# ╚══════════════════════════════════════════╝
# Mở camera → đếm ngược 5 giây → chụp ảnh → so sánh → in kết quả

# Dùng: python 04_recognize.py
#       python 04_recognize.py --loop       (nhận diện liên tục, không dừng)
#       python 04_recognize.py --countdown 3
# """
import cv2
import pickle
import time
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from datetime import datetime

from deepface import DeepFace
from config import (
    EMBED_FILE, MODEL_NAME, DETECTOR,
    THRESHOLD, CAMERA_INDEX, RECOGNIZE_COUNTDOWN,
    FRAME_WIDTH, FRAME_HEIGHT, LOG_DIR
)


# ── Load database ──────────────────────────────────────────────────────────

def load_db() -> dict:
    if not EMBED_FILE.exists():
        raise FileNotFoundError(
            f'Chưa có embedding database!\n'
            f'Chạy trước: python 03_train.py'
        )
    with open(EMBED_FILE, 'rb') as f:
        db = pickle.load(f)
    print(f'✅ Đã load DB: {len(db)} người — {list(db.keys())}')
    return db


# ── Nhận diện ─────────────────────────────────────────────────────────────

def recognize(img_path: str, db: dict) -> tuple[str, float, float]:
    """
    Nhận diện khuôn mặt trong ảnh.
    Trả về (tên, confidence%, cosine_distance)
    """
    try:
        result = DeepFace.represent(
            img_path         = img_path,
            model_name       = MODEL_NAME,
            detector_backend = DETECTOR,
            enforce_detection= False,
            align            = True,
        )
    except Exception as e:
        return 'NO_FACE', 0.0, 1.0

    if not result:
        return 'NO_FACE', 0.0, 1.0

    query_emb = np.array(result[0]['embedding'], dtype=np.float32)
    # Normalize
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm

    best_name = 'UNKNOWN'
    best_dist = float('inf')

    for person, data in db.items():
        dist = cosine(query_emb, data['mean'])
        if dist < best_dist:
            best_dist = dist
            best_name = person

    if best_dist > THRESHOLD:
        return 'UNKNOWN', 0.0, best_dist

    confidence = round((1.0 - best_dist) * 100, 1)
    return best_name, confidence, best_dist


# ── Log kết quả ───────────────────────────────────────────────────────────

def log_result(name: str, confidence: float, img_path: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f'recognize_{datetime.now():%Y%m%d}.txt'
    line = f'{datetime.now():%H:%M:%S}  {name:<25}  {confidence:5.1f}%  {img_path}\n'
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line)


# ── UI helpers ─────────────────────────────────────────────────────────────

def draw_countdown(frame, remaining: float, total: int):
    h, w = frame.shape[:2]

    # Làm tối 4 góc (vignette nhẹ)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Vòng tròn đếm ngược
    cx, cy  = w // 2, h // 2
    radius  = 80
    angle   = int(360 * (1 - remaining / total))
    cv2.circle(frame, (cx, cy), radius + 4, (40, 40, 40), 8)
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle,
                (0, 210, 120), 8)
    cv2.putText(frame, str(int(remaining) + 1), (cx - 25, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

    cv2.putText(frame, 'Dung yen nhin vao camera...', (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

    # Thanh tiến trình dưới
    bar_w   = w - 40
    fill_w  = int(bar_w * (1 - remaining / total))
    cv2.rectangle(frame, (20, h - 20), (20 + bar_w, h - 8), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 20), (20 + fill_w, h - 8), (0, 210, 120), -1)

    return frame


def draw_result(frame, name: str, confidence: float, dist: float):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if name == 'NO_FACE':
        color = (0, 100, 255)
        label = 'Khong phat hien khuon mat'
        sub   = 'Thu lai: nhin thang vao camera'
    elif name == 'UNKNOWN':
        color = (0, 60, 220)
        label = 'KHONG NHAN DIEN DUOC'
        sub   = f'Distance={dist:.3f} > threshold={THRESHOLD}'
    else:
        color = (0, 200, 80)
        label = name.replace('_', ' ').upper()
        sub   = f'Do tin cay: {confidence:.1f}%  |  dist={dist:.3f}'

    cv2.putText(frame, label, (20, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, sub, (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, 'SPACE = chup lai  |  Q = thoat', (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    return frame


# ── Main loop ──────────────────────────────────────────────────────────────

def run_recognition(loop_mode: bool = False,
                    countdown: int = RECOGNIZE_COUNTDOWN):
    """
    Mở camera, đếm ngược, chụp ảnh, nhận diện, hiển thị kết quả.
    loop_mode=True → tự động chụp lại sau khi hiện kết quả
    """
    db  = load_db()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f'❌ Không mở được camera (index={CAMERA_INDEX})')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    TMP_IMG = '/tmp/_face_capture.jpg'
    state   = 'countdown'   # countdown → captured → result → (loop: countdown)
    cd_start   = time.time()
    result_frame = None
    last_name = last_conf = last_dist = None

    print(f'📷 Camera bật | Đếm ngược {countdown}s')
    print('Nhấn Q để thoát | SPACE để chụp lại')
    print('─' * 40)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now     = time.time()
            display = frame.copy()

            if state == 'countdown':
                elapsed   = now - cd_start
                remaining = max(0.0, countdown - elapsed)
                display   = draw_countdown(display, remaining, countdown)

                if elapsed >= countdown:
                    # Chụp ảnh tại thời điểm này
                    cv2.imwrite(TMP_IMG, frame)
                    state = 'processing'
                    print('📸 Đang nhận diện...', end=' ', flush=True)

            elif state == 'processing':
                # Hiển thị frame "loading" trong khi xử lý
                cv2.putText(display, 'Dang xu ly...', (40, display.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
                cv2.imshow('Face Recognition', display)
                cv2.waitKey(1)

                # Nhận diện (có thể mất 1-3s lần đầu load model)
                name, conf, dist = recognize(TMP_IMG, db)
                last_name, last_conf, last_dist = name, conf, dist
                log_result(name, conf, TMP_IMG)

                # In kết quả ra terminal
                if name == 'NO_FACE':
                    print('❌ Không phát hiện khuôn mặt')
                elif name == 'UNKNOWN':
                    print(f'❓ Không nhận diện được (dist={dist:.3f})')
                else:
                    print(f'✅ {name} — {conf:.1f}% (dist={dist:.3f})')

                result_frame = frame.copy()
                state = 'result'
                result_show_time = time.time()
                continue

            elif state == 'result':
                display = result_frame.copy()
                display = draw_result(display, last_name, last_conf, last_dist)

                # Auto-loop sau 4 giây
                if loop_mode and (time.time() - result_show_time) > 4.0:
                    state    = 'countdown'
                    cd_start = time.time()

            cv2.imshow('Face Recognition', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print('\n👋 Thoát')
                break
            elif key == ord(' '):
                # Chụp lại ngay
                state    = 'countdown'
                cd_start = time.time()
                print('🔄 Chụp lại...')

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nhận diện khuôn mặt')
    parser.add_argument('--loop',      action='store_true',
                        help='Tự động chụp lại liên tục')
    parser.add_argument('--countdown', type=int, default=RECOGNIZE_COUNTDOWN,
                        help=f'Giây đếm ngược (mặc định={RECOGNIZE_COUNTDOWN})')
    args = parser.parse_args()

    run_recognition(loop_mode=args.loop, countdown=args.countdown)
