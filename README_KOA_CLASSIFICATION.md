# Klasifikasi Tingkat Keparahan Knee Osteoarthritis (KOA)

Repositori ini berisi notebook eksperimen untuk klasifikasi tingkat keparahan **Knee Osteoarthritis (KOA)** berdasarkan citra radiografi lutut. Pipeline utama menggunakan pra-pemrosesan citra, ekstraksi fitur CNN, standardisasi fitur, seleksi fitur menggunakan Genetic Algorithm (GA), serta model klasifikasi Support Vector Machine (SVM) dan Random Forest (RF).

## 1. Ringkasan Pipeline

Pipeline eksperimen secara umum terdiri dari:

1. Membaca dataset citra radiografi lutut.
2. Melakukan pra-pemrosesan citra:
   - konversi warna BGR ke RGB,
   - cropping ROI area sendi lutut jika tersedia pada notebook,
   - resize citra menjadi 224×224 piksel,
   - peningkatan kontras menggunakan CLAHE,
   - normalisasi input sesuai kebutuhan model CNN.
3. Membagi data menjadi train, validation, dan test secara stratified.
4. Menjalankan skenario penyeimbangan data, sesuai notebook:
   - imbalance data,
   - balancing to minority class,
   - balancing to majority class,
   - balancing to 1000 images per class,
   - SMOTE pada level fitur,
   - custom/reference CNN sebagai skenario pembanding.
5. Melatih CNN atau VGG19 sebagai feature extractor.
6. Mengekstraksi 200 fitur bottleneck.
7. Melakukan standardisasi fitur menggunakan `StandardScaler`.
8. Melakukan seleksi fitur menggunakan Genetic Algorithm.
9. Melatih classifier SVM dan Random Forest.
10. Mengevaluasi model menggunakan accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC, dan classification report.
11. Menyimpan model, scaler, hasil evaluasi, visualisasi, dan running log.

## 2. Struktur Dataset

Dataset utama disimpan di Google Drive:

```text
/content/drive/MyDrive/archive
```

Saat dijalankan di Google Colab, dataset dapat disalin ke storage lokal Colab agar proses training dan ekstraksi fitur lebih cepat:

```text
/content/archive
```

### 2.1 Struktur Dataset untuk Notebook Skenario Utama

Notebook skenario utama dapat membaca data berdasarkan folder kelas. Struktur yang disarankan:

```text
archive/
├── 0/
│   ├── image_001.png
│   └── ...
├── 1/
│   ├── image_001.png
│   └── ...
├── 2/
├── 3/
└── 4/
```

Keterangan label:

```text
0 = Grade 0 / Normal
1 = Grade 1 / Doubtful
2 = Grade 2 / Mild
3 = Grade 3 / Moderate
4 = Grade 4 / Severe
```

Beberapa notebook juga dapat membaca struktur folder yang lebih kompleks selama nama folder kelas masih mengandung label `0`, `1`, `2`, `3`, atau `4`.

### 2.2 Struktur Dataset untuk Custom CNN

Notebook `skenario_custom_cnn.ipynb` menggunakan struktur data yang sudah dipisahkan menjadi `train`, `val`, dan `test`:

```text
archive/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
└── test/
    ├── 0/
    ├── 1/
    ├── 2/
    ├── 3/
    └── 4/
```

## 3. Konfigurasi Google Colab

Pada bagian awal notebook, pastikan konfigurasi path dibuat seperti berikut:

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import shutil

DRIVE_DATASET_DIR = "/content/drive/MyDrive/archive"
DATASET_DIR       = "/content/archive"

if not Path(DATASET_DIR).exists():
    print("Menyalin dataset dari Google Drive ke local Colab...")
    shutil.copytree(DRIVE_DATASET_DIR, DATASET_DIR)
else:
    print("Dataset local sudah tersedia:", DATASET_DIR)
```

Penyalinan ke `/content/archive` disarankan karena membaca file langsung dari Google Drive biasanya lebih lambat dibanding storage lokal Colab.

## 4. Instalasi Dependency

Jika menjalankan di Google Colab, sebagian besar library seperti TensorFlow, NumPy, Pandas, Scikit-learn, dan Matplotlib biasanya sudah tersedia. Namun, untuk memastikan dependency lengkap, jalankan:

```bash
pip install -r requirements.txt
```

Jika terjadi konflik versi TensorFlow di Colab, gunakan runtime baru kemudian jalankan ulang instalasi. Setelah instalasi TensorFlow, lakukan restart runtime bila diminta.

## 5. Daftar Notebook Eksperimen

| Notebook | Fungsi |
|---|---|
| `skenario_imbalance_data_AUGMENTED_READY_RUNNING_RUNTIME_LOG...ipynb` | Eksperimen data tidak seimbang dengan pipeline fitur CNN, GA, SVM, dan RF |
| `skenario_minority_balancing_AUGMENTED_READY_RUNNING_RUNTIME_LOG...ipynb` | Eksperimen balancing training set ke jumlah kelas minoritas |
| `skenario_major_balancing_AUGMENTED_READY_RUNNING_RUNTIME_LOG...ipynb` | Eksperimen balancing training set ke jumlah kelas mayoritas |
| `skenario_balanced_1000_per_class_AUGMENTED_READY_RUNNING_RUNTIME_LOG...ipynb` | Eksperimen balancing training set menjadi 1000 citra per kelas |
| `skenario_imbalance_data_SMOTE_FEATURE_LEVEL_READY_RUNNING...ipynb` | Eksperimen SMOTE pada level fitur setelah ekstraksi fitur CNN |
| `koa_final_monitoring_adjusted_reference_cnn...ipynb` | Notebook pembanding untuk reference/custom CNN dengan monitoring training |
| `skenario_custom_cnn.ipynb` | Notebook custom/reference CNN dengan skenario balancing 1000 per class dan augmentasi |

## 6. Hyperparameter Tuning

Notebook versi Colab hyperparameter test menggunakan tuning berbasis `RandomizedSearchCV`.

Contoh ruang pencarian SVM:

```python
SVM_PARAM_DIST = {
    "C": [0.1, 1, 10, 50, 70, 100],
    "gamma": ["scale", "auto", 0.001, 0.01],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "class_weight": [None, "balanced"],
}
```

Contoh ruang pencarian Random Forest:

```python
RF_PARAM_DIST = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 4, 8, 12],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "bootstrap": [True],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "warm_start": [False, True],
    "n_jobs": [1],
}
```

Scoring utama yang disarankan:

```python
scoring = "roc_auc_ovr"
```

Metrik `roc_auc_ovr` digunakan karena penelitian ini merupakan klasifikasi multikelas. Pendekatan One-vs-Rest menghitung kemampuan model membedakan setiap grade terhadap seluruh grade lainnya, kemudian merata-ratakan skor AUC secara macro.

## 7. Output Eksperimen

Setelah notebook dijalankan, output utama akan disimpan ke folder model, umumnya:

```text
/content/drive/MyDrive/skripsi/model_final
```

Output yang biasanya dihasilkan:

```text
model_final/
├── *.h5 / *.keras                  # model CNN terbaik
├── *.pkl                           # classifier, scaler, indeks fitur GA
├── laporan_proses/                 # running log dan hasil proses
├── visual_outputs/                 # grafik training, scaler, ROC, CM
├── classification_report_*.csv     # laporan evaluasi
├── summary_metrics_*.csv           # ringkasan metrik
└── runtime_log_*.csv               # catatan waktu setiap proses
```

## 8. Cara Menjalankan Notebook

1. Upload notebook ke Google Colab.
2. Aktifkan GPU:
   - `Runtime` → `Change runtime type` → `GPU`.
3. Pastikan dataset tersedia di:
   ```text
   /content/drive/MyDrive/archive
   ```
4. Jalankan cell konfigurasi path.
5. Jalankan semua cell secara berurutan.
6. Periksa output evaluasi:
   - classification report,
   - confusion matrix,
   - ROC curve,
   - AUC per kelas,
   - summary metrics,
   - runtime log.

## 9. Catatan Penting

- Penyeimbangan data hanya dilakukan pada data training.
- Data validation dan test tidak boleh diberi augmentasi atau balancing agar evaluasi tetap objektif.
- `StandardScaler` harus di-fit hanya pada data training, lalu parameter mean dan standar deviasi yang sama digunakan untuk validation dan test.
- Jika preprocessing berubah, misalnya menambahkan cropping ROI sebelum resize, maka model harus dilatih ulang agar fitur, scaler, GA, dan classifier konsisten dengan input baru.
- Jika menggunakan SMOTE, SMOTE diterapkan pada level fitur setelah ekstraksi fitur, bukan langsung pada citra.
- Untuk laporan skripsi, metrik utama yang disarankan adalah macro ROC-AUC, weighted ROC-AUC, macro F1-score, weighted F1-score, confusion matrix, dan classification report per kelas.

## 10. Troubleshooting

### Dataset tidak ditemukan

Pastikan path Google Drive benar:

```python
DRIVE_DATASET_DIR = "/content/drive/MyDrive/archive"
```

Cek isi folder:

```python
import os
print(os.listdir(DRIVE_DATASET_DIR))
```

### GPU tidak aktif

Cek GPU:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

Jika kosong, aktifkan GPU dari menu runtime Colab.

### Error `predict_proba`

Untuk SVM dengan ROC-AUC, pastikan:

```python
SVC(probability=True)
```

### Error key metrik seperti `auc_val`

Gunakan nama key yang sesuai dengan output evaluasi terbaru, misalnya:

```python
auc_val_macro
auc_val_weighted
auc_test_macro
auc_test_weighted
```

### Runtime terlalu lama

Kurangi jumlah iterasi tuning:

```python
SVM_RANDOM_SEARCH_N_ITER = 10
RF_RANDOM_SEARCH_N_ITER = 15
CLASSIFIER_CV_FOLDS = 3
```

## 11. Informasi Penelitian

Judul penelitian:

```text
Optimasi Seleksi Fitur CNN Menggunakan Genetic Algorithm
untuk Klasifikasi Tingkat Keparahan Knee Osteoarthritis
```

Peneliti:

```text
Christian Marcello Dwisusanto
Program Studi S-1 Sistem Informasi
Universitas Airlangga
```
