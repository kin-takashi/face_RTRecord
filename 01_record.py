# """
# ╔══════════════════════════════════════╗
# ║  MODULE 01 — Quay video khuôn mặt   ║
# ╚══════════════════════════════════════╝
# Dùng: python 01_record.py --name nguyen_van_a
#       python 01_record.py --name tran_thi_b --duration 15
#"""
import cv2
import time
import argparse
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

from config import (
    VIDEO_DIR, CAMERA_INDEX,
    RECORD_DURATION_SEC, FRAME_WIDTH, FRAME_HEIGHT
)


def draw_ui(frame, elapsed, duration, name, state):
    """Vẽ countdown và hướng dẫn lên frame."""
    h, w = frame.shape[:2]
    remaining = max(0, duration - elapsed)

    # Overlay mờ phía trên
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Tên người
    cv2.putText(frame, f'Dang quay: {name}', (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if state == 'countdown':
        cv2.putText(frame, f'Bat dau trong: {int(remaining)+1}s', (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    elif state == 'recording':
        # Chấm đỏ nhấp nháy
        if int(elapsed * 2) % 2 == 0:
            cv2.circle(frame, (w - 40, 40), 12, (0, 0, 255), -1)
        cv2.putText(frame, f'REC  {remaining:.1f}s', (w - 160, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Thanh tiến trình
        bar_w = w - 40
        progress = elapsed / duration
        cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 30), (20 + int(bar_w * progress), h - 15),
                      (0, 180, 80), -1)

    # Hướng dẫn
    hints = [
        'Nhin thang > Xoay trai/phai > Nguc len/cui xuong > Tu nhien',
    ]
    cv2.putText(frame, hints[0], (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return frame


def record_video(name: str, duration: int = RECORD_DURATION_SEC,
                 camera: int = CAMERA_INDEX) -> str | None:
    """
    Quay video khuôn mặt và lưu vào thư mục videos/.
    Trả về đường dẫn file video, hoặc None nếu thất bại.
    """
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f'❌ Không mở được camera (index={camera})')
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam  = cap.get(cv2.CAP_PROP_FPS) or 30.0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = VIDEO_DIR / f'{name}_{timestamp}.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_cam, (actual_w, actual_h))

    print(f'📷 Camera: {actual_w}x{actual_h} @ {fps_cam:.0f}fps')
    print(f'🎬 Sẽ lưu: {out_path}')
    print('─' * 50)
    print('Hướng dẫn quay:')
    print('  3s — Nhìn thẳng vào camera')
    print('  3s — Xoay mặt trái rồi phải (nhẹ nhàng)')
    print('  3s — Ngước lên rồi cúi xuống')
    print('  3s — Tự nhiên, thử cười / bình thường')
    print('─' * 50)
    print('Nhấn SPACE để bắt đầu quay | Q để thoát')

    state       = 'waiting'   # waiting → recording → done
    rec_start   = None
    frames_written = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()

            if state == 'waiting':
                draw_ui(frame, 0, duration, name, 'countdown')
                cv2.putText(frame, 'Nhan SPACE de bat dau', (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

            elif state == 'recording':
                elapsed = now - rec_start
                draw_ui(frame, elapsed, duration, name, 'recording')
                writer.write(frame)
                frames_written += 1
                if elapsed >= duration:
                    state = 'done'
                    print(f'\n✅ Quay xong! {frames_written} frames (~{frames_written/fps_cam:.1f}s)')
                    break

            cv2.imshow('Record Face — Nhan Q de thoat', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print('\n⚠️  Người dùng thoát sớm')
                break
            elif key == ord(' ') and state == 'waiting':
                state     = 'recording'
                rec_start = time.time()
                print('🔴 Đang quay...')

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    if frames_written < 30:
        print('⚠️  Video quá ngắn — thử lại')
        out_path.unlink(missing_ok=True)
        return None

    print(f'💾 Video đã lưu: {out_path} ({out_path.stat().st_size/1024:.0f} KB)')
    return str(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quay video khuôn mặt')
    parser.add_argument('--name',     help='Tên người (vd: nguyen_van_a)')
    parser.add_argument('--duration', type=int, default=RECORD_DURATION_SEC,
                        help=f'Thời gian quay (giây, mặc định={RECORD_DURATION_SEC})')
    parser.add_argument('--camera',   type=int, default=CAMERA_INDEX)
    args = parser.parse_args()

    if not args.name:
        args.name = input('👤 Nhập tên người (vd: hieu): ').strip()
        if not args.name:
            print('❌ Tên không được để trống!')
            exit(1)

    result = record_video(args.name, args.duration, args.camera)
    if result:
        print(f'\n➡️  Bước tiếp: python 02_extract.py --name {args.name} --video "{result}"')
