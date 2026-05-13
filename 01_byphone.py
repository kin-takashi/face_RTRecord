import cv2
import numpy as np
import requests
import time
import os
from datetime import datetime

os.makedirs("videos", exist_ok=True)



# ================== CẤU HÌNH ==================
# THAY ĐỊA CHỈ IP CỦA BẠN VÀO ĐÂY (thường kết thúc bằng :8080/video)
IP_WEBCAM_URL = "https://192.168.50.124:8080/video"   # ←←← THAY ĐỊA CHỈ NÀY

RECORD_SECONDS = 15          # Thởi lượng ghi video (10-15 giây)
FPS = 30                    # Frame rate (có thể chỉnh 15-30)
FRAME_WIDTH = 640            # Độ phân giải (càng cao càng nặng)
FRAME_HEIGHT = 480
# =============================================


def record_video_by_phone(name: str, record_seconds: int = RECORD_SECONDS) -> str | None:
    """
    Ghi video từ camera điện thoại (IP Webcam) và lưu vào videos/{name}_{timestamp}.mp4.
    Trả về đường dẫn file video, hoặc None nếu thất bại.
    """
    if not name:
        print("❌ Tên không được để trống!")
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join("videos", f"{name}_{timestamp}.mp4")

    print(f"📱 Đang chuẩn bị ghi video cho: {name}")
    print(f"💾 File lưu: {output_file}")

    # Mở stream từ camera điện thoại
    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    if not cap.isOpened():
        print("❌ Không thể kết nối camera điện thoại!")
        print("Kiểm tra lại:")
        print("1. Điện thoại và máy tính cùng WiFi")
        print("2. Đã Start server trong app IP Webcam")
        print("3. Địa chỉ IP_WEBCAM_URL có đúng không?")
        return None

    # Thiết lập độ phân giải
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Tạo đối tượng ghi video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    print(f"✅ Đã kết nối camera điện thoại. Bắt đầu ghi video...")
    print(f"⏱️  Ghi trong {record_seconds} giây...")

    start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️  Không nhận được frame, thử lại...")
                time.sleep(0.1)
                continue

            # Resize frame về kích thước cố định
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # Ghi frame vào video
            out.write(frame)
            
            # Hiển thị preview
            cv2.imshow("Phone Camera - Đang ghi video", frame)
            
            frame_count += 1
            
            # Dừng sau RECORD_SECONDS giây
            if time.time() - start_time > record_seconds:
                break

            # Nhấn 'q' để dừng sớm
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⛔ Dừng theo lệnh ngườii dùng.")
                break

    finally:
        # Giải phóng tài nguyên
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"✅ Hoàn tất! Video đã lưu: {output_file}")
    print(f"📊 Tổng số frame ghi được: {frame_count}")
    return output_file


if __name__ == '__main__':
    name = input("👤 Nhập tên người (vd: hieu): ").strip()
    if not name:
        print("❌ Tên không được để trống!")
        exit(1)
    
    result = record_video_by_phone(name)
    if result:
        print(f"\n➡️  Bước tiếp: python 02_extract.py --name {name} --video \"{result}\"")

