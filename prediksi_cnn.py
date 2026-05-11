"""
prediksi_cnn.py
Real-time Face Recognition menggunakan model CNN (train_cnn.py)
"""

import cv2, numpy as np, pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# Config — HARUS SAMA dengan train_cnn.py
# ============================================================
MODEL_PATH           = 'model/cnn_model.keras'
LABEL_ENCODER_PATH   = 'model/label_encoder_cnn.pickle'
IMG_SIZE             = 96
CONFIDENCE_THRESHOLD = 0.70   # 70% — ubah dengan +/-
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

# CLAHE — SAMA seperti saat training
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_face(face_gray):
    face_eq  = clahe.apply(face_gray)
    face_rgb = cv2.cvtColor(face_eq, cv2.COLOR_GRAY2BGR)
    face_rs  = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_f32 = face_rs.astype("float32")
    face_pre = preprocess_input(face_f32)
    return np.expand_dims(face_pre, axis=0)           # (1, 96, 96, 3)

# ============================================================
# Haar Cascade
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

# Buffer voting per wajah (anti-flicker)
name_buffer  = deque(maxlen=BUFFER_SIZE)
proba_buffer = deque(maxlen=BUFFER_SIZE)

print("[INFO] Tekan 'q' keluar | '+'/'-' ubah threshold")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Tambah padding di sekitar wajah
        pad = int(min(w, h) * 0.12)
        x1, y1 = max(0, x-pad),              max(0, y-pad)
        x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
        face_roi = gray[y1:y2, x1:x2]

        # Prediksi
        face_in = preprocess_face(face_roi)
        preds   = model.predict(face_in, verbose=0)[0]
        j       = np.argmax(preds)
        proba   = float(preds[j])
        name    = le.classes_[j]

        # Simpan ke buffer
        name_buffer.append(name)
        proba_buffer.append(proba)

        # Voting stabilitas
        stable_name  = Counter(name_buffer).most_common(1)[0][0]
        stable_proba = float(np.mean(
            [p for n, p in zip(name_buffer, proba_buffer) if n == stable_name]
        ))

        # Tentukan label
        if stable_proba >= CONFIDENCE_THRESHOLD:
            label = f"{stable_name} ({stable_proba*100:.1f}%)"
            color = (0, 220, 0)
        else:
            label = f"Unknown ({stable_proba*100:.1f}%)"
            color = (0, 50, 220)

        # Gambar kotak
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label dengan background
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1
        )
        cv2.rectangle(frame, (x, y-th-12), (x+tw+6, y), color, -1)
        cv2.putText(frame, label, (x+3, y-5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 1)

        # Debug top-3 di terminal
        top3 = np.argsort(preds)[::-1][:3]
        info = "  |  ".join(
            [f"{le.classes_[i]}: {preds[i]*100:.1f}%" for i in top3]
        )
        print(f"\r[PRED] {info}    ", end="")

    # HUD threshold
    cv2.putText(frame,
                f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%  (+/- untuk ubah)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

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
