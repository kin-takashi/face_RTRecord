"""
Cấu hình chung cho toàn bộ hệ thống
Chỉnh sửa file này — không cần sửa các module khác
"""
from pathlib import Path

# ── Đường dẫn ──────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATASET_DIR    = BASE_DIR / 'dataset'      # Ảnh khuôn mặt từng người
DATAPREP_DIR     = BASE_DIR / 'predata'    # Ảnh đã chuẩn hóa (face aligned)
VIDEO_DIR      = BASE_DIR / 'videos'       # Video gốc
EMBED_DIR      = BASE_DIR / 'embeddings'   # File .pkl embedding
LOG_DIR        = BASE_DIR / 'logs'

# ── Model DeepFace ──────────────────────────────
MODEL_NAME   = 'ArcFace'      # ArcFace | Facenet512 | VGG-Face
DETECTOR     = 'retinaface'       # opencv (nhanh, CPU) | retinaface (chính xác hơn, chậm hơn)
# Lưu ý: retinaface cần thêm pip install retina-face
EMBED_FILE   = EMBED_DIR / f'db_{MODEL_NAME.lower()}.pkl'

# ── Ngưỡng nhận diện ───────────────────────────
THRESHOLD    = 0.40   # Cosine distance. Thấp hơn = chặt hơn
#   0.30 → rất chặt (ít false positive, có thể miss)
#   0.40 → cân bằng ← dùng mặc định
#   0.50 → rộng hơn (dễ nhận diện, tăng nhầm)

# ── Thu thập ảnh từ video ──────────────────────
CAPTURE_EVERY_N_FRAMES = 8    # Lấy 1 frame mỗi N frame
MAX_IMAGES_PER_PERSON  = 100
MIN_IMAGES_PER_PERSON  = 60

# ── Chất lượng ảnh ─────────────────────────────
MIN_FACE_SIZE          = 80   # pixel
BLUR_THRESHOLD         = 100.0
MIN_BRIGHTNESS         = 40
MAX_BRIGHTNESS         = 230
DUPLICATE_THRESHOLD    = 8    # pHash hamming distance

# ── Camera ─────────────────────────────────────
CAMERA_INDEX           = 0
RECORD_DURATION_SEC    = 12   # Thời gian quay video (giây)
RECOGNIZE_COUNTDOWN    = 5    # Đếm ngược trước khi chụp (giây)
FRAME_WIDTH            = 1280
FRAME_HEIGHT           = 720

# Tạo thư mục nếu chưa có
for d in [DATASET_DIR, VIDEO_DIR, EMBED_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
