[INFO] Memuat dataset (RGB + face detector)...
  [OK] Faza Humairah                          | dipakai: 105 | di-skip (tanpa wajah): 0
  [OK] Jabbal Akbar                           | dipakai: 105 | di-skip (tanpa wajah): 0
  [OK] Muhammad Alif                          | dipakai: 104 | di-skip (tanpa wajah): 1
  [OK] Muhammad Heikal Fasya                  | dipakai: 105 | di-skip (tanpa wajah): 0
  [OK] Muhammad Rizki Arta Maulana            | dipakai: 105 | di-skip (tanpa wajah): 0
  [OK] Nabila Balqis                          | dipakai: 105 | di-skip (tanpa wajah): 0
  [OK] Natasya Babulrhaiyan                   | dipakai: 104 | di-skip (tanpa wajah): 1
  [OK] Nur Fadhillah Zulfi                    | dipakai: 105 | di-skip (tanpa wajah): 0
Traceback (most recent call last):
  File "D:\FaceAttend_VC\train_cnn.py", line 168, in <module>
    data, labels = load_data(DATASET_PATH)
  File "D:\FaceAttend_VC\train_cnn.py", line 136, in load_data
    box = detect_face_box(img_rgb)
  File "D:\FaceAttend_VC\train_cnn.py", line 71, in detect_face_box
    results = mtcnn_detector.detect_faces(img_rgb)
  File "D:\FaceAttend_VC\venv\lib\site-packages\mtcnn\mtcnn.py", line 159, in detect_faces
    bboxes_batch = stage(bboxes_batch=bboxes_batch, images_normalized=images_normalized, images_oshapes=images_oshapes, **kwargs)
  File "D:\FaceAttend_VC\venv\lib\site-packages\mtcnn\stages\stage_pnet.py", line 99, in __call__
    bboxes_nms = [smart_nms_from_bboxes(b, threshold=nms_pnet1, method="union", initial_sort=False) for b in bboxes_batch_upscaled]
  File "D:\FaceAttend_VC\venv\lib\site-packages\mtcnn\stages\stage_pnet.py", line 99, in <listcomp>
    bboxes_nms = [smart_nms_from_bboxes(b, threshold=nms_pnet1, method="union", initial_sort=False) for b in bboxes_batch_upscaled]
  File "D:\FaceAttend_VC\venv\lib\site-packages\mtcnn\utils\bboxes.py", line 251, in smart_nms_from_bboxes
    target_iou = iou(target_bboxes[:, columns_bbox], method=method)
  File "D:\FaceAttend_VC\venv\lib\site-packages\mtcnn\utils\bboxes.py", line 157, in iou
    area_union = area_bboxes[:, None] + area_bboxes[None, :] - area_inter
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.42 GiB for an array with shape (13829, 13829) and data type float64
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1778598583.229060   22176 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.  
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1778598593.573034   22176 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.  
I0000 00:00:1778598602.712237   22176 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
[INFO] MTCNN tersedia -> dipakai untuk deteksi wajah real-time.
[INFO] Memuat model CNN...
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
[INFO] Kelas (19): [np.str_('Faza Humairah'), np.str_('Jabbal Akbar'), np.str_('Muhammad Alif'), np.str_('Muhammad Heikal Fasya'), np.str_('Muhammad Rizki Arta Maulana'), np.str_('Nabila Balqis'), np.str_('Natasya Babulrhaiyan'), np.str_('Nur Fadhillah Zulfi'), np.str_('Putri Al Violy'), np.str_('Rahmat Isma Hidayat'), np.str_('Riski Maulani'), np.str_('Salesya Al Fatila'), np.str_('Suci Wildani Rizka'), np.str_('Syariqul Husni'), np.str_('Syifaurrahman'), np.str_('Tasya Anisa'), np.str_('Urfi Shanda'), np.str_('Vivi Jumilia Hikmah'), np.str_('Zuyyin Zafirah')]
[INFO] Menyalakan kamera...
[INFO] Tekan 'q' keluar | '+'/'-' ubah threshold
[PRED] Rahmat Isma Hidayat: 58.5%  |  Syifaurrahman: 5.1%  |  Syariqul Husni: 4.9%    W0000 00:00:1778598614.153774   22176 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Incompatible shapes: [0,48,48,3] vs. [1,1,1,32]
[PRED] Salesya Al Fatila: 42.7%  |  Rahmat Isma Hidayat: 20.5%  |  Riski Maulani: 10.2%    W0000 00:00:1778598618.288190   22176 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Incompatible shapes: [0,48,48,3] vs. [1,1,1,32]
[PRED] Salesya Al Fatila: 43.5%  |  Rahmat Isma Hidayat: 19.6%  |  Syariqul Husni: 7.4%    W0000 00:00:1778598630.501797   22176 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Incompatible shapes: [0,48,48,3] vs. [1,1,1,32]
[PRED] Rahmat Isma Hidayat: 35.4%  |  Salesya Al Fatila: 24.2%  |  Natasya Babulrhaiyan: 6.0%    W0000 00:00:1778598633.369296   22176 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Incompatible shapes: [0,48,48,3] vs. [1,1,1,32]
W0000 00:00:1778598637.659987   22176 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Incompatible shapes: [0,48,48,3] vs. [1,1,1,32]
W0000 00:00:1778598645.812209   22176 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Incompatible shapes: [0,48,48,3] vs. [1,1,1,32]
W0000 00:00:1778598659.896270   22176 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Incompatible shapes: [0,48,48,3] vs. [1,1,1,32]
[PRED] Rahmat Isma Hidayat: 53.3%  |  Salesya Al Fatila: 15.3%  |  Nur Fadhillah Zulfi: 8.2%    W0000 00:00:1778598700.862333   22176 local_rendezvous.cc:412] Local rendezvous is aborting with status: INVALID_ARGUMENT: Incompatible shapes: [0,48,48,3] vs. [1,1,1,32]
[PRED] Urfi Shanda: 76.2%  |  Natasya Babulrhaiyan: 3.7%  |  Nur Fadhillah Zulfi: 2.9%         
[INFO] Kamera dimatikan.
