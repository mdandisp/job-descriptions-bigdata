# Global Job Descriptions Data Analysis & Machine Learning

Proyek ini merupakan implementasi Praktikum Big Data dari Kelompok 7 (UPN Veteran Jakarta 2026). Kami mengolah dan menganalisis dataset "Global Job Descriptions" berukuran raksasa (1.7 GB) untuk mengekstraksi insight dunia kerja, memprediksi jabatan (NLP), memprediksi kompensasi gaji (Regresi), serta melakukan klasterisasi segmentasi perusahaan.

## 👥 Tim Pengembang (Kelompok 7)
- **Akmal Taufiqurrahman** (2310511018)
- **Wisnu Chandra Mukti Wibowo** (2310511026)
- **Muhammad Dandi Setiawan Putra** (2310511035)

## 📌 Deskripsi Proyek
Mengingat ukuran dataset yang melebihi batas tradisional Microsoft Excel (ratusan ribu baris, 1.7 GB), proyek ini dibangun dengan library performa tinggi **Polars** untuk fase pemrosesan data (ETL) dan **Pandas/Scikit-Learn** untuk fase eksplorasi (EDA) dan Machine Learning.

Tiga pilar utama proyek ini:
1. **Pembersihan Data (ETL)**: Mem-parsing string gaji seperti `"$50K-$100K"` menjadi metrik kuantitatif dan menangani *missing values*.
2. **Exploratory Data Analysis (EDA)**: Menyingkap distribusi pasar kerja berdasarkan peran, tren perekrutan, kualifikasi akademik, hingga deteksi pola data.
3. **Machine Learning Pipeline**:
   - **Klasifikasi Teks**: Pemanfaatan *TF-IDF* untuk memprediksi profesi murni dari teks deskripsi pekerjaan.
   - **Regresi Harga**: Prediksi nilai kompensasi (*Salary*) berdasarkan berbagai faktor industri.
   - **Klasterisasi (Unsupervised)**: Segmentasi korporasi menggunakan algoritma *K-Means*.

## 📂 Struktur Repositori
- `kode_praktikum.ipynb` : Versi interaktif (Jupyter Notebook) yang disarankan untuk dibaca. Memuat seluruh proses secara runut beserta output grafik dan sel penjelasan.
- `kode_praktikum.py` : Skrip Python penuh bagi Anda yang ingin menjalankan seluruh pipeline di terminal atau server tanpa environment interaktif.
- `convert.py` & `update_png_path.py` : Skrip *utility* tambahan.
- `output/` : Direktori berisi seluruh hasil ekstraksi plot/grafik analitik beresolusi 16:9.

## 🚀 Cara Menjalankan Secara Lokal

**1. Persiapan Dataset**
Karena batasan ukuran file di GitHub, Anda harus mengunduh dataset secara mandiri:
- Unduh `job_descriptions.csv` dari [Kaggle (Ravindra Singh Rana)](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset)
- Letakkan file tersebut di akar (root) repositori ini (sejajar dengan file `.ipynb`).

**2. Instalasi Dependensi**
Pastikan Anda menggunakan Python versi terbaru dan menginstal pustaka yang dibutuhkan:
```bash
pip install polars pyarrow pandas numpy scikit-learn matplotlib seaborn
```

**3. Eksekusi**
Anda memiliki dua opsi eksekusi:
- **Opsi A (Direkomendasikan):** Buka file `kode_praktikum.ipynb` menggunakan VS Code / Jupyter IDE, lalu klik "Run All". Anda dapat melihat interaksi data step-by-step.
- **Opsi B:** Jalankan lewat terminal dengan perintah `python kode_praktikum.py`. Seluruh output gambar akan diekspor langsung ke folder `output/`.
