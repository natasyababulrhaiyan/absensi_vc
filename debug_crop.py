"""
debug_crop.py
Visual debug crop wajah: bandingkan crop dari dataset vs crop dari webcam.

Cara pakai:
  python debug_crop.py            # jalankan dua-duanya
  python debug_crop.py dataset    # hanya dump dari dataset
  python debug_crop.py webcam     # hanya dump dari webcam (tekan 's' simpan)

Hasil:
  debug/dataset_crops/<nama>.jpg
  debug/webcam_crops/webcam_001.jpg ...

Buka 2 folder itu, bandingkan: apakah crop dari dataset dan webcam terlihat
similar (zoom, padding, sudut, lighting). Kalau beda jauh -> sumber masalah.
"""

import os, sys, cv2, numpy as np

# ============================================================
# Config (HARUS SAMA dengan train_cnn.py)
# ============================================================
DATASET_PATH      = 'Dataset/Dataset_wajah'
IMG_SIZE          = 160
MAX_DETECTOR_DIM  = 640

DEBUG_DIR_DATASET = 'debug/dataset_crops'
DEBUG_DIR_WEBCAM  = 'debug/webcam_crops'
os.makedirs(DEBUG_DIR_DATASET, exist_ok=True)
os.makedirs(DEBUG_DIR_WEBCAM, exist_ok=True)

# ============================================================
# Detector (logika persis seperti train_cnn.py)
# ============================================================
try:
    from mtcnn import MTCNN
    HAS_MTCNN = True
    mtcnn_detector = MTCNN()
    print("[INFO] MTCNN tersedia.")
except Exception:
    HAS_MTCNN = False
    print("[WARN] MTCNN tidak terinstall. Pakai Haar Cascade.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def resize_for_detector(img_rgb):
    h, w = img_rgb.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DETECTOR_DIM:
        return img_rgb, 1.0
    scale = MAX_DETECTOR_DIM / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def detect_face_box_haar(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda r: r[2] * r[3]))

def detect_face_box(img_rgb):
    h_img, w_img = img_rgb.shape[:2]
    if HAS_MTCNN:
        detector_img, scale = resize_for_detector(img_rgb)
        try:
            results = mtcnn_detector.detect_faces(detector_img)
        except Exception:
            return detect_face_box_haar(img_rgb)
        if results:
            confident = [r for r in results if r.get('confidence', 0) >= 0.80]
            candidates = confident or results
            candidates.sort(
                key=lambda r: r.get('confidence', 0) * r['box'][2] * r['box'][3],
                reverse=True
            )
            x, y, w, h = candidates[0]['box']
            x = int(round(x / scale))
            y = int(round(y / scale))
            w = int(round(w / scale))
            h = int(round(h / scale))
            x, y = max(0, x), max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            if w > 0 and h > 0:
                return (x, y, w, h)
        return detect_face_box_haar(img_rgb)
    return detect_face_box_haar(img_rgb)

# ============================================================
# Crop helper (padding SAMA dengan train_cnn.py)
# ============================================================
def crop_with_pad(img_rgb, box, pad_ratio=0.12):
    x, y, w, h = box
    pad = int(min(w, h) * pad_ratio)
    H, W = img_rgb.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return img_rgb[y1:y2, x1:x2]

# ============================================================
# Mode 1: Dump 1 crop per orang dari dataset
# ============================================================
def dump_dataset_crops():
    print(f"\n[INFO] Dumping crop dataset -> {DEBUG_DIR_DATASET}")
    if not os.path.isdir(DATASET_PATH):
        print(f"[ERROR] Folder dataset tidak ada: {DATASET_PATH}")
        return

    for person_name in sorted(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_path):
            continue

        saved = False
        for fname in sorted(os.listdir(person_path)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_bgr = cv2.imread(os.path.join(person_path, fname))
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            box = detect_face_box(img_rgb)
            if box is None:
                continue

            face_rgb = crop_with_pad(img_rgb, box)
            if face_rgb.size == 0:
                continue
            face_rs  = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_AREA)
            face_bgr = cv2.cvtColor(face_rs, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(DEBUG_DIR_DATASET, f"{person_name}.jpg")
            cv2.imwrite(out_path, face_bgr)
            print(f"  [OK] {person_name:<40} -> {fname}")
            saved = True
            break

        if not saved:
            print(f"  [!!] {person_name:<40} - tidak ada wajah terdeteksi")

    print(f"[INFO] Selesai. Buka folder: {DEBUG_DIR_DATASET}")

# ============================================================
# Mode 2: Capture crop dari webcam
# ============================================================
def dump_webcam_crops():
    print(f"\n[INFO] Webcam mode. Tekan 's' = simpan crop, 'q' = keluar.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    counter = len([f for f in os.listdir(DEBUG_DIR_WEBCAM)
                   if f.startswith('webcam_')])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box = detect_face_box(frame_rgb)

        display = frame.copy()
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 0), 2)

        cv2.putText(display, "s = simpan crop  |  q = keluar",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.imshow("Debug Webcam Crop", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if box is None:
                print("  [SKIP] Tidak ada wajah.")
                continue
            face_rgb = crop_with_pad(frame_rgb, box)
            if face_rgb.size == 0:
                continue
            face_rs  = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_AREA)
            face_bgr = cv2.cvtColor(face_rs, cv2.COLOR_RGB2BGR)
            counter += 1
            out_path = os.path.join(DEBUG_DIR_WEBCAM,
                                    f"webcam_{counter:03d}.jpg")
            cv2.imwrite(out_path, face_bgr)
            print(f"  [OK] Simpan -> {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Selesai. Buka folder: {DEBUG_DIR_WEBCAM}")

# ============================================================
# Entry
# ============================================================
if __name__ == '__main__':
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'

    if mode in ('dataset', 'all'):
        dump_dataset_crops()
    if mode in ('webcam', 'all'):
        dump_webcam_crops()
    if mode not in ('dataset', 'webcam', 'all'):
        print(f"[ERROR] Mode tidak dikenal: {mode}")
        print("Pakai: python debug_crop.py [dataset|webcam|all]")
