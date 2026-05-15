"""
capture_realtime.py
Capture foto wajah realtime dari webcam laptop untuk MENAMBAH data training
yang berasal dari "distribusi yang sama dengan inferensi".

Hasilnya disimpan ke:
    Dataset/Dataset_wajah/<NAMA>/realtime_NNN.jpg

Cara pakai (cmd / PowerShell):
    python capture_realtime.py --name "Natasya Babulrhaiyan" --n 30
    python capture_realtime.py --name "Muhammad Alif"        --n 50 --out Dataset/Dataset_wajah

Tombol di window kamera:
    SPACE = mulai / jeda perekaman
    Q     = keluar
"""

import argparse
import os
import time

import cv2

try:
    from mtcnn import MTCNN
    HAS_MTCNN = True
except Exception:
    HAS_MTCNN = False


def capture_realtime(name: str,
                     n_samples: int = 30,
                     output_dir: str = "Dataset/Dataset_wajah",
                     delay_sec: float = 0.30,
                     min_face_ratio: float = 0.15) -> int:
    """Capture `n_samples` frame yang mengandung wajah ke folder per orang.

    - Hanya menyimpan frame ketika MTCNN mendeteksi 1 wajah dgn conf >= 0.9
      DAN ukuran wajah memenuhi `min_face_ratio` (proporsi diagonal frame).
    - Frame disimpan UTUH (bukan hasil crop) supaya konsisten dgn dataset lama
      yang akan diproses ulang oleh detect_face() di notebook.
    """
    person_dir = os.path.join(output_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    # cari index berikutnya supaya tidak overwrite file lama
    existing = [f for f in os.listdir(person_dir)
                if f.startswith("realtime_") and f.endswith(".jpg")]
    start_idx = 0
    for f in existing:
        try:
            idx = int(f.replace("realtime_", "").replace(".jpg", ""))
            start_idx = max(start_idx, idx + 1)
        except ValueError:
            pass

    detector = MTCNN() if HAS_MTCNN else None
    if detector is None:
        print("[WARN] MTCNN tidak tersedia. Fallback ke Haar Cascade.")
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa buka kamera.")
        return 0

    saved = 0
    capturing = False
    last_save = 0.0
    fh, fw = None, None

    print(f"[INFO] Capture untuk : {name}")
    print(f"[INFO] Target sampel : {n_samples}")
    print(f"[INFO] Folder simpan : {person_dir}")
    print( "[INFO] SPACE = mulai/jeda, Q = keluar")
    print( "[INFO] Saran: variasikan POSE, JARAK, dan CAHAYA selama capture")

    while saved < n_samples:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame gagal dibaca.")
            break

        if fh is None:
            fh, fw = frame.shape[:2]
            min_face_px = int(min_face_ratio * min(fh, fw))

        # ===== deteksi wajah utk feedback visual =====
        face_ok = False
        box_to_draw = None

        if detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb)
            results = [r for r in results if r["confidence"] >= 0.90]
            results.sort(key=lambda r: r["box"][2] * r["box"][3], reverse=True)
            if results:
                x, y, w, h = results[0]["box"]
                box_to_draw = (max(0, x), max(0, y), w, h)
                if min(w, h) >= min_face_px:
                    face_ok = True
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            if len(faces):
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                box_to_draw = (x, y, w, h)
                if min(w, h) >= min_face_px:
                    face_ok = True

        # ===== overlay HUD =====
        disp = frame.copy()
        if box_to_draw is not None:
            x, y, w, h = box_to_draw
            col = (0, 220, 0) if face_ok else (0, 165, 255)
            cv2.rectangle(disp, (x, y), (x + w, y + h), col, 2)

        status_text = "RECORDING" if capturing else "PAUSED"
        status_col = (0, 220, 0) if capturing else (0, 165, 255)
        cv2.putText(disp, f"{status_text}   {saved}/{n_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2)
        cv2.putText(disp, f"name: {name}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(disp, "SPACE=start/pause  Q=quit  variasikan pose/jarak/cahaya",
                    (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        cv2.imshow("Capture Realtime", disp)

        # ===== save kalau memenuhi syarat & tidak terlalu cepat =====
        now = time.time()
        if capturing and face_ok and (now - last_save) >= delay_sec:
            idx = start_idx + saved
            fname = os.path.join(person_dir, f"realtime_{idx:03d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
            last_save = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            capturing = not capturing
            if capturing:
                print("[INFO] Recording...")
            else:
                print("[INFO] Paused.")

    cap.release()
    cv2.destroyAllWindows()

    print(f"[OK] Tersimpan {saved} frame ke '{person_dir}'.")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Capture wajah realtime dari webcam ke folder dataset."
    )
    parser.add_argument("--name", required=False, type=str, default=None,
                        help="Nama orang (akan jadi nama subfolder). Wajib jika --loop tidak dipakai.")
    parser.add_argument("--n", type=int, default=30,
                        help="Jumlah frame target per orang (default 30).")
    parser.add_argument("--out", type=str, default="Dataset/Dataset_wajah",
                        help="Root folder dataset.")
    parser.add_argument("--delay", type=float, default=0.30,
                        help="Jeda detik antar simpan frame.")
    parser.add_argument("--min-face", type=float, default=0.15,
                        help="Min rasio wajah thd diagonal frame (0..1).")
    parser.add_argument("--loop", action="store_true",
                        help="Mode interaktif: setelah selesai 1 orang, tanya nama berikutnya.")
    args = parser.parse_args()

    if args.loop:
        print("=" * 60)
        print("MODE LOOP - capture banyak orang berturut-turut")
        print("Ketik nama orang lalu ENTER. Kosongkan + ENTER untuk keluar.")
        print("=" * 60)
        while True:
            name = input("\n[?] Nama orang berikutnya (kosong = selesai): ").strip()
            if not name:
                print("[INFO] Loop dihentikan.")
                break
            try:
                n_input = input(f"[?] Jumlah sampel untuk '{name}' (default {args.n}): ").strip()
                n_samples = int(n_input) if n_input else args.n
            except ValueError:
                n_samples = args.n
            capture_realtime(
                name=name,
                n_samples=n_samples,
                output_dir=args.out,
                delay_sec=args.delay,
                min_face_ratio=args.min_face,
            )
        return

    if not args.name:
        parser.error("--name wajib diisi (atau gunakan --loop untuk mode interaktif).")

    capture_realtime(
        name=args.name,
        n_samples=args.n,
        output_dir=args.out,
        delay_sec=args.delay,
        min_face_ratio=args.min_face,
    )


if __name__ == "__main__":
    main()
