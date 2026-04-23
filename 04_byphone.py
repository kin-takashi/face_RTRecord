"""
╔══════════════════════════════════════════╗
║  MODULE 04_byphone — Nhận diện từ PHONE  ║
╚══════════════════════════════════════════╝
Mở camera PHONE (IP Webcam) → đếm ngược 5s → chụp ảnh → so sánh → in kết quả

SETUP PHONE:
1. Cài app 'IP Webcam' trên Android
2. Start server → copy URL (ví dụ: https://192.168.50.124:8080/video)
3. THAY URL bên dưới cho đúng IP của bạn
4. Phone và PC cùng WiFi

Dùng: python 04_byphone.py [--countdown 3]
   → Mở cam → countdown → capture → đóng cam → RESULT → END
"""

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
    THRESHOLD, RECOGNIZE_COUNTDOWN,
    FRAME_WIDTH, FRAME_HEIGHT, LOG_DIR
)

# ── IP Webcam từ phone (THAY IP NÀY!) ───────────────────────
IP_WEBCAM_URL = "https://192.168.50.124:8080/video"  # ←←← THAY IP PHONE CỦA BẠN

# ── Load database ───────────────────────────────────────────

def load_db() -> dict:
    if not EMBED_FILE.exists():
        raise FileNotFoundError(
            f'Chưa có embedding database!\\n'
            f'Chạy trước: python 03_train.py'
        )
    with open(EMBED_FILE, 'rb') as f:
        db = pickle.load(f)
    print(f'✅ Đã load DB: {len(db)} người — {list(db.keys())}')
    return db

# ── Nhận diện ──────────────────────────────────────────────

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

# ── Log kết quả ─────────────────────────────────────────────

def log_result(name: str, confidence: float, img_path: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f'recognize_phone_{datetime.now():%Y%m%d}.txt'
    line = f'{datetime.now():%H:%M:%S}  {name:<25}  {confidence:5.1f}%  {img_path}\\n'
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line)

# ── UI helpers ──────────────────────────────────────────────

def draw_countdown(frame, remaining: float, total: int):
    h, w = frame.shape[:2]

    # Làm tối 4 góc
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Vòng tròn đếm ngược
    cx, cy  = w // 2, h // 2
    radius  = 80
    angle   = int(360 * (1 - remaining / total))
    cv2.circle(frame, (cx, cy), radius + 4, (40, 40, 40), 8)
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, (0, 210, 120), 8)
    cv2.putText(frame, str(int(remaining) + 1), (cx - 25, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

    cv2.putText(frame, '📱 PHONE CAMERA | Dung yen...', (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

    # Thanh tiến trình
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
        label = '📱 Khong phat hien khuon mat'
        sub   = 'Nhìn rõ vào phone camera'
    elif name == 'UNKNOWN':
        color = (0, 60, 220)
        label = 'KHONG NHAN DIEN DUOC'
        sub   = f'Distance={dist:.3f} > {THRESHOLD}'
    else:
        color = (0, 200, 80)
        label = name.replace('_', ' ').upper()
        sub   = f'📱 Confidence: {confidence:.1f}%  |  dist={dist:.3f}'

    cv2.putText(frame, label, (20, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, sub, (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, 'SPACE = chup lai  |  Q = thoat', (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    return frame

# ── Main loop ─────────────────────────────────────────────────

def run_recognition(loop_mode: bool = False, countdown: int = RECOGNIZE_COUNTDOWN):
    """
    Mở IP Webcam → countdown → capture → recognize
    """
    db = load_db()
    
    print(f'📱 Kết nối PHONE camera: {IP_WEBCAM_URL}')
    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    
    if not cap.isOpened():
        print('❌ Không kết nối được phone camera!')
        print('Kiểm tra:')
        print('1. Phone chạy IP Webcam, cùng WiFi')
        print(f'2. URL đúng: {IP_WEBCAM_URL}')
        print('3. Thử mở URL trên browser trước')
        return

    # Set resolution (có thể giảm nếu lag)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    TMP_IMG = 'temp_phone_capture.jpg'  # Local temp file
    state   = 'countdown'
    cd_start = time.time()
    result_frame = None
    last_name = last_conf = last_dist = None

    print(f'✅ Phone camera OK | Countdown {countdown}s')
    print('Nhấn Q thoát | SPACE chụp lại')
    print('─' * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('⚠️ Mất kết nối stream, retry...')
                time.sleep(0.1)
                continue

            now = time.time()
            display = frame.copy()

            if state == 'countdown':
                elapsed = now - cd_start
                remaining = max(0.0, countdown - elapsed)
                display = draw_countdown(display, remaining, countdown)

                if elapsed >= countdown:
                    cv2.imwrite(TMP_IMG, frame)
                    state = 'processing'
                    print('📸 Phone capture → recognizing... ', end='', flush=True)

            elif state == 'processing':
                cv2.putText(display, 'Processing... (DeepFace)', (40, display.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
                cv2.imshow('Phone Face Recognition', display)
                cv2.waitKey(1)

                name, conf, dist = recognize(TMP_IMG, db)
                last_name, last_conf, last_dist = name, conf, dist
                log_result(name, conf, TMP_IMG)

                if name == 'NO_FACE':
                    print('❌ No face detected')
                elif name == 'UNKNOWN':
                    print(f'❓ Unknown (dist={dist:.3f})')
                else:
                    print(f'✅ {name} — {conf:.1f}% confidence')

                result_frame = frame.copy()
                state = 'result'
                result_show_time = time.time()
                continue

            elif state == 'result':
                display = result_frame.copy()
                display = draw_result(display, last_name, last_conf, last_dist)

                if loop_mode and (time.time() - result_show_time) > 4.0:
                    state = 'countdown'
                    cd_start = time.time()

            cv2.imshow('Phone Face Recognition', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print('\\n👋 Exit')
                break
            elif key == ord(' '):
                state = 'countdown'
                cd_start = time.time()
                print('🔄 Recapture...')

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phone Camera Recognition')
    parser.add_argument('--countdown', type=int, default=RECOGNIZE_COUNTDOWN,
                        help=f'Seconds wait before capture (default={RECOGNIZE_COUNTDOWN})')
    args = parser.parse_args()

    run_recognition(countdown=args.countdown)

