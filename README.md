# 🎯 Face Attendance — Local Setup

## Cấu trúc project
```
face_attendance/
├── config.py          ← Cấu hình toàn bộ hệ thống (chỉnh ở đây)
├── 01_record.py       ← Quay video khuôn mặt
├── 02_extract.py      ← Trích xuất ảnh từ video
├── 03_train.py        ← Build embedding database
├── 04_recognize.py    ← Nhận diện khuôn mặt
├── requirements.txt
├── dataset/           ← Ảnh khuôn mặt (tự tạo)
├── embeddings/        ← File .pkl (tự tạo)
├── videos/            ← Video gốc (tự tạo)
└── logs/              ← Log nhận diện (tự tạo)
```

---

## 🚀 Cài đặt lần đầu
```bash
pip install -r requirements.txt
```

---

## 📋 Quy trình đầy đủ

### Bước 1 — Quay video (12 giây)
```bash
python 01_record.py --name nguyen_van_a
```
> Nhấn **SPACE** để bắt đầu quay. Di chuyển mặt: thẳng → trái/phải → lên/xuống.

### Bước 2 — Trích xuất khuôn mặt (~80-100 ảnh)
```bash
python 02_extract.py --name nguyen_van_a
# Hoặc chỉ định video cụ thể:
python 02_extract.py --name nguyen_van_a --video videos/nguyen_van_a_20240115.mp4
```

### Lặp lại Bước 1-2 cho từng người
```bash
python 01_record.py --name tran_thi_b
python 02_extract.py --name tran_thi_b

python 01_record.py --name le_van_c
python 02_extract.py --name le_van_c
```

### Bước 3 — Build embedding database
```bash
python 03_train.py
```
> Lần đầu chạy sẽ tải model ArcFace (~500MB) — chỉ tải 1 lần.

### Bước 4 — Nhận diện
```bash
# Chụp 1 lần sau 5 giây đếm ngược
python 04_recognize.py

# Nhận diện liên tục (loop)
python 04_recognize.py --loop

# Đếm ngược 3 giây
python 04_recognize.py --countdown 3
```

---

## ⚙️ Tuỳ chỉnh trong config.py

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `DETECTOR` | `'opencv'` | `'opencv'` (nhanh) hoặc `'retinaface'` (chính xác hơn) |
| `THRESHOLD` | `0.40` | Ngưỡng nhận diện. Giảm nếu nhận nhầm, tăng nếu hay UNKNOWN |
| `MODEL_NAME` | `'ArcFace'` | `'ArcFace'` / `'Facenet512'` / `'VGG-Face'` |
| `RECOGNIZE_COUNTDOWN` | `5` | Giây đếm ngược trước khi chụp |
| `MAX_IMAGES_PER_PERSON` | `100` | Số ảnh tối đa lấy từ 1 video |

---

## 🔧 Thêm người mới (không cần train lại từ đầu)
```bash
python 01_record.py --name ten_nguoi_moi
python 02_extract.py --name ten_nguoi_moi
python 03_train.py --name ten_nguoi_moi   # Chỉ update người mới
```

## 🗑️ Xây lại DB từ đầu
```bash
python 03_train.py --rebuild
```
