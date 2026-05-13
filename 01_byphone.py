import cv2
import numpy as np
import time
import os
os.makedirs("videos", exist_ok=True)



# ================== CẤU HÌNH ==================
# THAY ĐỊA CHỈ IP CỦA BẠN VÀO ĐÂY (thường kết thúc bằng :8080/video)
IP_WEBCAM_URL = "https://192.168.50.124:8080/video"   # ←←← THAY ĐỊA CHỈ NÀY

RECORD_SECONDS = 12          # Thời lượng ghi video (10-15 giây)
FPS = 20                     # Frame rate (có thể chỉnh 15-30)
FRAME_WIDTH = 640            # Độ phân giải (càng cao càng nặng)
FRAME_HEIGHT = 480
# =============================================

# Tìm số file lớn nhất để đặt tên tiếp theo
def get_next_filename():
    i = 0
    while True:
        filename = os.path.join("videos", f"record_{i}.mp4")
        if not os.path.exists(filename):
            return filename
        i += 1

output_file = get_next_filename()
print(f"Đang chuẩn bị ghi video vào thư mục videos/ với tên: {os.path.basename(output_file)}")

# Mở stream từ camera điện thoại
cap = cv2.VideoCapture(IP_WEBCAM_URL)

if not cap.isOpened():
    print("❌ Không thể kết nối camera điện thoại!")
    print("Kiểm tra lại:")
    print("1. Điện thoại và máy tính cùng WiFi")
    print("2. Đã Start server trong app IP Webcam")
    print("3. Địa chỉ IP_WEBCAM_URL có đúng không?")
    exit()

# Thiết lập độ phân giải
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Tạo đối tượng ghi video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

print("✅ Đã kết nối camera điện thoại. Bắt đầu ghi video...")
print(f"Ghi trong {RECORD_SECONDS} giây...")

start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("⚠️ Không nhận được frame, thử lại...")
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
    if time.time() - start_time > RECORD_SECONDS:
        break

    # Nhấn 'q' để dừng sớm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Dừng theo lệnh người dùng.")
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Hoàn tất! Video đã lưu: {output_file}")
print(f"Tổng số frame ghi được: {frame_count}")