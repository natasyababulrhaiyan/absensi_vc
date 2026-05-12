"""
train_cnn.py  v3
Training Face Recognition dengan CNN + Transfer Learning (MobileNetV2)

Perbaikan dari v2 (point 1-8):
  1. RGB ASLI (bukan grayscale -> fake RGB)
  2. IMG_SIZE 160 (lebih cocok dengan MobileNetV2 pretrained)
  3. Skip gambar yang tidak terdeteksi wajah (tidak fallback ke full image)
  4. BatchNorm di base MobileNetV2 selalu di-freeze saat fine-tuning
  5. Augmentasi DULU, preprocess_input BELAKANGAN (lewat preprocessing_function)
  6. class_weight='balanced' untuk kebal terhadap dataset tidak seimbang
  7. MTCNN sebagai detector utama (fallback Haar Cascade jika tidak ada)
  8. Head model disederhanakan: Dense -> ReLU -> Dropout (tanpa BN, dengan L2)
"""

import os, cv2, numpy as np, pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2

# ============================================================
# Config  HARUS SAMA dengan prediksi_cnn.py
# ============================================================
DATASET_PATH       = 'Dataset/Dataset_wajah'
MODEL_SAVE_PATH    = 'model/cnn_model.keras'
LABEL_ENCODER_PATH = 'model/label_encoder_cnn.pickle'
IMG_SIZE           = 160     # MobileNetV2 jauh lebih baik di >=160
BATCH_SIZE         = 32      # Dataset 100/orang sudah cukup untuk batch besar
EPOCHS             = 50
MAX_DETECTOR_DIM   = 640     # Batasi input MTCNN supaya tidak boros RAM

os.makedirs('model', exist_ok=True)

# ============================================================
# Face Detector: MTCNN (akurat) dengan fallback Haar Cascade
# ============================================================
try:
    from mtcnn import MTCNN
    HAS_MTCNN = True
    print("[INFO] MTCNN tersedia -> akan dipakai untuk crop wajah training.")
    mtcnn_detector = MTCNN()
except Exception:
    HAS_MTCNN = False
    print("[WARN] MTCNN tidak terinstall. Pakai Haar Cascade (kurang akurat).")
    print("[WARN] Untuk akurasi terbaik, install: pip install mtcnn")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def resize_for_detector(img_rgb):
    """Resize sementara untuk detector; box nanti dimapping balik ke ukuran asli."""
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
    """
    Return (x, y, w, h) wajah TERBESAR di gambar RGB,
    atau None jika tidak ada wajah terdeteksi.
    """
    h_img, w_img = img_rgb.shape[:2]

    if HAS_MTCNN:
        detector_img, scale = resize_for_detector(img_rgb)
        try:
            results = mtcnn_detector.detect_faces(detector_img)
        except Exception as e:
            print(f"[WARN] MTCNN gagal di 1 gambar ({type(e).__name__}). Fallback Haar.")
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
        else:
            return detect_face_box_haar(img_rgb)

        x, y = max(0, x), max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)

    return detect_face_box_haar(img_rgb)

# ============================================================
# CLAHE pada gambar BERWARNA (channel L pada LAB) -> aman warna
# ============================================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def clahe_color(img_rgb):
    """Equalize kontras tanpa merusak warna (CLAHE pada channel L LAB)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def preprocess_face(face_rgb):
    """CLAHE -> resize. TIDAK pakai preprocess_input di sini.
    preprocess_input dipanggil belakangan oleh ImageDataGenerator
    (preprocessing_function) supaya augmentasi terjadi di range [0,255]."""
    face_eq = clahe_color(face_rgb)
    face_rs = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    return face_rs.astype('float32')   # tetap di range [0, 255]

# ============================================================
# Load Dataset
# ============================================================
def load_data(dataset_path):
    data, labels = [], []
    total_skipped = 0

    print("[INFO] Memuat dataset (RGB + face detector)...")
    for person_name in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        count, skipped = 0, 0
        for fname in os.listdir(person_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            img_bgr = cv2.imread(os.path.join(person_path, fname))
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            box = detect_face_box(img_rgb)

            if box is None:
                skipped += 1   # JANGAN dipakai (tidak fallback ke full image)
                continue

            (x, y, w, h) = box
            pad = int(min(w, h) * 0.12)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_rgb.shape[1], x + w + pad)
            y2 = min(img_rgb.shape[0], y + h + pad)
            face_roi = img_rgb[y1:y2, x1:x2]

            if face_roi.size == 0:
                skipped += 1
                continue

            data.append(preprocess_face(face_roi))
            labels.append(person_name)
            count += 1

        total_skipped += skipped
        flag = "[OK]" if count > 0 else "[!!]"
        print(f"  {flag} {person_name:<38} | dipakai: {count:3d} | "
              f"di-skip (tanpa wajah): {skipped}")

    print(f"\n[INFO] Total dipakai: {len(data)} gambar | "
          f"total di-skip: {total_skipped}")
    return np.array(data, dtype='float32'), np.array(labels)


data, labels = load_data(DATASET_PATH)
print(f"[INFO] Shape data: {data.shape}  (range [0,255] sebelum augmentasi)")

# ============================================================
# Encode Label
# ============================================================
le = LabelEncoder()
y  = le.fit_transform(labels)
Y  = to_categorical(y)
NUM_CLASSES = len(le.classes_)

print(f"[INFO] Jumlah kelas: {NUM_CLASSES} -> {list(le.classes_)}")

with open(LABEL_ENCODER_PATH, 'wb') as f:
    pickle.dump(le, f)
print("[INFO] Label encoder disimpan.")

# ============================================================
# Split Train / Val
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    data, Y, test_size=0.2, stratify=Y, random_state=42
)
print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)}")

# ============================================================
# Class Weight (kebal terhadap kelas tidak seimbang)
# ============================================================
y_train_int = y_train.argmax(axis=1)
cw_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weight_dict = {i: w for i, w in enumerate(cw_array)}
print(f"[INFO] class_weight: {class_weight_dict}")

# ============================================================
# Data Augmentation (v4: lebih agresif untuk generalisasi ke webcam)
#   - Augmentasi geometri di range [0,255]
#   - Tambahan augmentasi piksel (blur, noise, random erasing)
#     dilakukan di `aug_then_preprocess` lalu preprocess_input
#   - horizontal_flip=True (standar di face recognition modern)
#   - Validation diproses manual TANPA augmentasi
# ============================================================
def aug_then_preprocess(img):
    """
    Dipanggil oleh ImageDataGenerator paling akhir.
    img: float32 array (H, W, 3), range [0, 255] setelah aug geometri.
    Tambah: Gaussian blur, noise, random erasing -> paksa model belajar
    fitur wajah yang invariant ke kondisi pencahayaan/kamera, dan tidak
    bergantung pada 1 region (rambut/baju/mulut saja).
    """
    img = img.astype('float32')

    if np.random.random() < 0.30:
        ksize = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    if np.random.random() < 0.25:
        noise = np.random.normal(0.0, 8.0, img.shape).astype('float32')
        img = np.clip(img + noise, 0.0, 255.0)

    if np.random.random() < 0.25:
        h, w = img.shape[:2]
        eh = np.random.randint(max(1, h // 8), max(2, h // 4))
        ew = np.random.randint(max(1, w // 8), max(2, w // 4))
        ey = np.random.randint(0, max(1, h - eh))
        ex = np.random.randint(0, max(1, w - ew))
        img[ey:ey + eh, ex:ex + ew] = np.random.uniform(
            0.0, 255.0, (eh, ew, img.shape[2])
        ).astype('float32')

    return preprocess_input(img)


datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    brightness_range=[0.7, 1.3],     # lebih lebar -> tahan kondisi kamera
    horizontal_flip=True,            # standar face recognition (FaceNet/ArcFace)
    fill_mode='nearest',
    preprocessing_function=aug_then_preprocess
)

X_val_pre = preprocess_input(X_val.copy())   # validation: langsung preprocess

# ============================================================
# Bangun Model CNN (MobileNetV2 sebagai feature extractor)
# Head disederhanakan: Dense -> ReLU -> Dropout (tanpa BN, dengan L2)
# ============================================================
print("[INFO] Membangun model...")
base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False   # Phase 1: hanya training head

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.02),
    metrics=['accuracy']
)
model.summary()

# ============================================================
# Phase 1: Training head saja (base MobileNetV2 frozen)
# ============================================================
print("\n[PHASE 1] Training classification head (base frozen)...")
callbacks_p1 = [
    EarlyStopping(monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1)
]

history1 = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
    validation_data=(X_val_pre, y_val),
    steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_p1
)

loss1, acc1 = model.evaluate(X_val_pre, y_val, verbose=0)
print(f"\n[Phase 1 Result] Val Accuracy: {acc1*100:.2f}%")

# ============================================================
# Phase 2: Fine-tuning  unfreeze 40 layer terakhir MobileNetV2
# PENTING: BatchNorm SELALU di-freeze (training=False) supaya
# running mean/var tidak rusak karena batch kecil.
# ============================================================
print("\n[PHASE 2] Fine-tuning 40 layer terakhir MobileNetV2...")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

frozen_bn = 0
for layer in base.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False
        frozen_bn += 1
print(f"[INFO] Total BatchNorm di base yang di-freeze: {frozen_bn}")

trainable_layers = sum(1 for l in base.layers if l.trainable)
print(f"[INFO] Trainable layer di base: {trainable_layers} dari {len(base.layers)}")

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.02),
    metrics=['accuracy']
)

callbacks_p2 = [
    EarlyStopping(monitor='val_accuracy', patience=10,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-8, verbose=1)
]

history2 = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
    validation_data=(X_val_pre, y_val),
    steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_p2
)

# ============================================================
# Evaluasi Final
# ============================================================
loss2, acc2 = model.evaluate(X_val_pre, y_val, verbose=0)
print(f"\n{'='*50}")
print(f"[Phase 1] Val Accuracy : {acc1*100:.2f}%")
print(f"[Phase 2] Val Accuracy : {acc2*100:.2f}%  <-- setelah fine-tune")
print(f"[Final]   Val Loss     : {loss2:.4f}")
print(f"[INFO]    Model disimpan: {MODEL_SAVE_PATH}")
print("[DONE] Selesai! Jalankan: python prediksi_cnn.py")
