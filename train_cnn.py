"""
train_cnn.py  v2
Training Face Recognition dengan CNN + Transfer Learning (MobileNetV2)
2 fase training:
  Phase 1 - Train head saja (base frozen)
  Phase 2 - Fine-tune 40 layer terakhir MobileNetV2 (LR sangat kecil)
Fix:
  - CLAHE preprocessing untuk normalisasi pencahayaan
  - Tidak ada horizontal_flip
  - Label smoothing mencegah model collapse ke 1 kelas
"""

import os, cv2, numpy as np, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

# ============================================================
# Config — HARUS SAMA dengan prediksi_cnn.py
# ============================================================
DATASET_PATH       = 'Dataset/Dataset_wajah'
MODEL_SAVE_PATH    = 'model/cnn_model.keras'
LABEL_ENCODER_PATH = 'model/label_encoder_cnn.pickle'
IMG_SIZE           = 96      # Input MobileNetV2
BATCH_SIZE         = 16
EPOCHS             = 50

os.makedirs('model', exist_ok=True)

# ============================================================
# Helper: CLAHE (normalisasi kontras/pencahayaan)
# ============================================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_face(face_gray):
    """Equalize histogram → resize → convert ke RGB → preprocess_input."""
    face_eq  = clahe.apply(face_gray)                          # CLAHE
    face_rgb = cv2.cvtColor(face_eq, cv2.COLOR_GRAY2BGR)
    face_rs  = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_f32 = face_rs.astype("float32")
    return preprocess_input(face_f32)                          # skala [-1,1]

# ============================================================
# Load Haar Cascade
# ============================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ============================================================
# Load Dataset
# ============================================================
def load_data(dataset_path):
    data, labels = [], []
    total_fallback = 0

    print("[INFO] Memuat dataset...")
    for person_name in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        count, fallback = 0, 0
        for fname in os.listdir(person_path):
            if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                continue
            img = cv2.imread(os.path.join(person_path, fname))
            if img is None:
                continue

            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
            )

            if len(faces) > 0:
                # Ambil wajah terbesar
                (x, y, w, h) = max(faces, key=lambda r: r[2]*r[3])
                pad = int(min(w, h) * 0.1)
                x1,y1 = max(0,x-pad), max(0,y-pad)
                x2,y2 = min(gray.shape[1],x+w+pad), min(gray.shape[0],y+h+pad)
                face_roi = gray[y1:y2, x1:x2]
            else:
                face_roi = gray   # fallback pakai gambar utuh
                fallback += 1

            processed = preprocess_face(face_roi)
            data.append(processed)
            labels.append(person_name)
            count += 1

        total_fallback += fallback
        flag = "[!]" if fallback > count * 0.3 else "[OK]"
        print(f"  {flag} {person_name:<38} | {count} gambar | fallback: {fallback}")

    print(f"\n[INFO] Total: {len(data)} gambar | "
          f"Fallback (tanpa deteksi): {total_fallback}")
    return np.array(data), np.array(labels)

data, labels = load_data(DATASET_PATH)
print(f"[INFO] Shape data: {data.shape}")

# ============================================================
# Encode Label
# ============================================================
le = LabelEncoder()
y  = le.fit_transform(labels)
Y  = to_categorical(y)
NUM_CLASSES = len(le.classes_)

print(f"[INFO] Jumlah kelas: {NUM_CLASSES} → {list(le.classes_)}")

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
# Data Augmentation (TANPA horizontal_flip!)
# ============================================================
datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    brightness_range=[0.75, 1.25],
    horizontal_flip=False,       # ← JANGAN flip wajah
    fill_mode='nearest'
)
datagen.fit(X_train)

# ============================================================
# Bangun Model CNN (MobileNetV2 sebagai feature extractor)
# ============================================================
print("[INFO] Membangun model...")
base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False   # freeze semua — hanya training head

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)

# Label smoothing mencegah model 100% confident ke 1 kelas
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

# ============================================================
# Phase 1: Training head saja (base MobileNetV2 frozen)
# ============================================================
print("\n[PHASE 1] Training classification head (base frozen)...")
callbacks_p1 = [
    EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH, monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=4, min_lr=1e-6, verbose=1
    )
]

history1 = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=callbacks_p1
)

loss1, acc1 = model.evaluate(X_val, y_val, verbose=0)
print(f"\n[Phase 1 Result] Val Accuracy: {acc1*100:.2f}%")

# ============================================================
# Phase 2: Fine-tuning — unfreeze 40 layer terakhir MobileNetV2
# Ini yang membuat model belajar fitur WAJAH, bukan fitur generik
# ============================================================
print("\n[PHASE 2] Fine-tuning 40 layer terakhir MobileNetV2...")
base.trainable = True
# Freeze semua kecuali 40 layer terakhir
for layer in base.layers[:-40]:
    layer.trainable = False

# Harus compile ulang setelah mengubah trainable
model.compile(
    optimizer=Adam(learning_rate=1e-5),   # LR sangat kecil!
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_p2 = [
    EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH, monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-8, verbose=1
    )
]

history2 = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=callbacks_p2
)

# ============================================================
# Evaluasi Final
# ============================================================
loss2, acc2 = model.evaluate(X_val, y_val, verbose=0)
print(f"\n{'='*50}")
print(f"[Phase 1] Val Accuracy : {acc1*100:.2f}%")
print(f"[Phase 2] Val Accuracy : {acc2*100:.2f}%  <-- setelah fine-tune")
print(f"[Final]   Val Loss     : {loss2:.4f}")
print(f"[INFO]    Model disimpan: {MODEL_SAVE_PATH}")
print("[DONE] Selesai! Jalankan: python prediksi_cnn.py")
