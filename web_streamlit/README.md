# KOA Web Classification

Website sederhana berbasis Streamlit untuk klasifikasi tingkat keparahan Knee Osteoarthritis dari citra radiografi lutut.

## Struktur Folder

Letakkan file model Anda di folder `models/`.

```
koa_web_streamlit/
├── app.py
├── requirements.txt
└── models/
    ├── best_rf_ga.pkl
    ├── scaler.pkl
    ├── selected_features_ga.npy
    └── vgg19_attention_extractor.h5
```

## File yang Dibutuhkan

1. `best_rf_ga.pkl`
   - Model klasifikasi akhir, misalnya Random Forest + GA atau SVM + GA.
2. `scaler.pkl`
   - StandardScaler yang sudah di-fit pada data training.
3. `selected_features_ga.npy`
   - Mask atau indeks fitur hasil Genetic Algorithm.
   - Jika model `.pkl` dilatih dengan 200 fitur penuh, file ini boleh tidak ada.
4. `vgg19_attention_extractor.h5` atau `vgg19_attention_extractor.keras`
   - Model Keras untuk ekstraksi fitur 200 dimensi dari citra.

Catatan penting: file `.pkl` classifier saja biasanya belum cukup, karena model tersebut tidak menerima citra mentah. Citra perlu diproses melalui extractor CNN, scaler, dan mask GA.

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Website

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

## Penyesuaian Normalisasi

Di `app.py`, cek:

```python
NORMALIZATION_MODE = "rescale_01"
```

Jika training memakai `img / 255.0`, biarkan `rescale_01`.
Jika training memakai `tf.keras.applications.vgg19.preprocess_input`, ubah menjadi `vgg19`.

## Catatan

Aplikasi ini merupakan prototipe sistem pendukung penelitian. Hasil prediksi tidak boleh digunakan sebagai pengganti diagnosis dokter spesialis radiologi.