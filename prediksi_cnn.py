"""
prediksi_cnn.py  v3
Real-time Face Recognition menggunakan model CNN (train_cnn.py)

Preprocessing HARUS SAMA dengan train_cnn.py:
  - Input RGB (bukan grayscale fake-RGB)
  - IMG_SIZE 160
  - CLAHE pada channel L (LAB color space)
  - preprocess_input MobileNetV2 dipanggil paling akhir
"""

import cv2, numpy as np, pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# Config  HARUS SAMA dengan train_cnn.py
# ============================================================
MODEL_PATH           = 'model/cnn_model.keras'
LABEL_ENCODER_PATH   = 'model/label_encoder_cnn.pickle'
IMG_SIZE             = 160
CONFIDENCE_THRESHOLD = 0.70   # 70%  ubah dengan +/-
BUFFER_SIZE          = 7      # frame voting untuk stabilitas

# ============================================================
# Load Model & Label Encoder
# ============================================================
print("[INFO] Memuat model CNN...")
try:
    model = load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    print(f"[INFO] Kelas ({len(le.classes_)}): {list(le.classes_)}")
except Exception as e:
    print(f"[ERROR] {e}")
    print("[ERROR] Jalankan dulu: python train_cnn.py")
    exit()

# ============================================================
# CLAHE pada gambar BERWARNA (channel L pada LAB)
# SAMA seperti saat training
# ============================================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def clahe_color(img_rgb):
    """Equalize kontras tanpa merusak warna."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def preprocess_face(face_rgb):
    """CLAHE -> resize -> float32 -> preprocess_input -> batch dim."""
    face_eq  = clahe_color(face_rgb)
    face_rs  = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_AREA)
    face_f32 = face_rs.astype('float32')
    face_pre = preprocess_input(face_f32)
    return np.expand_dims(face_pre, axis=0)   # (1, 160, 160, 3)

# ============================================================
# Haar Cascade (dipakai untuk real-time karena cepat)
# ============================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ============================================================
# Kamera
# ============================================================
print("[INFO] Menyalakan kamera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Kamera tidak bisa dibuka!")
    exit()

name_buffer  = deque(maxlen=BUFFER_SIZE)
proba_buffer = deque(maxlen=BUFFER_SIZE)

print("[INFO] Tekan 'q' keluar | '+'/'-' ubah threshold")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        pad = int(min(w, h) * 0.12)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        face_roi = frame_rgb[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        face_in = preprocess_face(face_roi)
        preds   = model.predict(face_in, verbose=0)[0]
        j       = int(np.argmax(preds))
        proba   = float(preds[j])
        name    = le.classes_[j]

        name_buffer.append(name)
        proba_buffer.append(proba)

        stable_name  = Counter(name_buffer).most_common(1)[0][0]
        stable_proba = float(np.mean(
            [p for n, p in zip(name_buffer, proba_buffer) if n == stable_name]
        ))

        if stable_proba >= CONFIDENCE_THRESHOLD:
            label = f"{stable_name} ({stable_proba*100:.1f}%)"
            color = (0, 220, 0)
        else:
            label = f"Unknown ({stable_proba*100:.1f}%)"
            color = (0, 50, 220)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1
        )
        cv2.rectangle(frame, (x, y - th - 12), (x + tw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

        top3 = np.argsort(preds)[::-1][:3]
        info = "  |  ".join(
            [f"{le.classes_[i]}: {preds[i]*100:.1f}%" for i in top3]
        )
        print(f"\r[PRED] {info}    ", end="")

    cv2.putText(frame,
                f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%  (+/- untuk ubah)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("FaceAttend VC - CNN", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('+'), ord('=')):
        CONFIDENCE_THRESHOLD = min(0.99, CONFIDENCE_THRESHOLD + 0.05)
        print(f"\n[INFO] Threshold naik: {CONFIDENCE_THRESHOLD:.2f}")
    elif key == ord('-'):
        CONFIDENCE_THRESHOLD = max(0.10, CONFIDENCE_THRESHOLD - 0.05)
        print(f"\n[INFO] Threshold turun: {CONFIDENCE_THRESHOLD:.2f}")

print("\n[INFO] Kamera dimatikan.")
cap.release()
cv2.destroyAllWindows()
