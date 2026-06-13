# Financial Customer Analytics Pipeline

Proyek *end-to-end Machine Learning* yang menggabungkan metode *Unsupervised Learning* (K-Means & PCA) untuk segmentasi nasabah dan *Supervised Learning* (Decision Tree & Random Forest Berbasis Tuning) untuk memprediksi profil aktivitas transaksi keuangan.

---

## Latar Belakang
Di era digitalisasi perbankan, institusi keuangan mengelola data transaksi berskala besar setiap harinya. Memahami profil perilaku nasabah sangat krusial untuk mendeteksi anomali aktivitas keuangan, mengoptimalkan sistem keamanan finansial, serta menyediakan layanan penawaran produk yang tepat sasaran. Pendekatan analisis konvensional sering kali gagal melihat pola perilaku yang tersembunyi jika data hanya dianalisis secara agregat. Oleh karena itu, proyek ini menerapkan pendekatan ilmu data (*data science*) terintegrasi guna mengelompokkan data aktivitas transaksi ke dalam klaster perilaku tertentu, kemudian membangun sistem klasifikasi prediktif untuk mengenali profil kelompok nasabah tersebut secara instan.

## Tujuan Proyek
1. **Customer Segmentation:** Mengelompokkan nasabah perbankan berdasarkan pola aktivitas, kebiasaan login, dan profil transaksi finansial menggunakan algoritma **K-Means Clustering**.
2. **Reduksi Dimensi:** Menyederhanakan kompleksitas multidimensi fitur menggunakan **Principal Component Analysis (PCA)** untuk efisiensi komputasi dan visualisasi sebaran klaster yang jelas.
3. **Pemodelan Klasifikasi Prediktif:** Membangun arsitektur model berbasis pohon keputusan (*Decision Tree*) dan ansambel (*Random Forest*) untuk mengklasifikasikan data nasabah ke dalam target klaster secara akurat.
4. **Optimasi Hyperparameter:** Melakukan pencarian parameter terbaik melalui eksperimen sistematis untuk memaksimalkan performa metrik klasifikasi dan mencegah *overfitting*.

## Dataset
Dataset yang digunakan mencakup **2.512 sampel data transaksi** yang merekam profil demografis perbankan serta detail teknis operasional nasabah. Fitur-fitur utama yang dianalisis meliputi:
- **Informasi Finansial:** `TransactionAmount` (Nilai transaksi keuangan) dan `AccountBalance` (Sisa saldo akun).
- **Profil Demografis:** `CustomerAge`, `AgeGroup` (Kategori Kelompok Usia: Muda, Dewasa, Tua), dan `CustomerOccupation` (Profesi seperti Doctor, Engineer, Student, Retired, dll).
- **Aktivitas Operasional:** `TransactionType` (Jenis transaksi: Credit / Debit), `Location` (Nama kota transaksi di AS), `Channel` (Kanal transaksi: ATM, Branch, Online), `TransactionDuration` (Durasi pengerjaan transaksi), dan `LoginAttempts` (Jumlah percobaan masuk ke akun).

Berkas keluaran data yang dikelola:
- `data/data_clustering.csv`: Data fitur hasil transformasi skala (*scaled*) dan prapemrosesan lengkap dengan kolom label `Target`.
- `data/data_clustering_inverse.csv`: Data hasil klastering yang dikembalikan ke skala nilai asli (*inverse transform*) untuk mempermudah interpretasi karakteristik asli nasabah di tiap segmen.

## Tahapan Analisis
Alur kerja komprehensif dipisahkan ke dalam dua notebook terstruktur:

### 1. Tahap Pembelajaran Tak Terarah (`financial_customer_clustering.ipynb`)
- **Prapemrosesan Data & Feature Engineering:** Transformasi variabel kategori (*Categorical Encoding*), penanganan nilai hilang jika ada, pembentukan fitur kelompok usia (`AgeGroup`), serta standarisasi fitur numerik menggunakan `StandardScaler`.
- **Reduksi Dimensi (PCA):** Mereduksi ruang dimensi fitur numerik menjadi komponen utama (`PCA1` dan `PCA2`) untuk mempermudah visualisasi sebaran spasial objek data.
- **Evaluasi Jumlah Klaster:** Penentuan nilai kelompok optimal ($K$) menggunakan visualisasi grafis metode siku (**Elbow Method** via `KElbowVisualizer`) serta perhitungan **Silhouette Score**.
- **Model Deployment & Ekspor Data:** Melatih model final K-Means, menyimpan objek model biner, menetapkan hasil label kelompok ke dalam kolom `Target`, serta mengekspor data hasil transformasi balik (*inverse*).

### 2. Tahap Pembelajaran Terarah (`financial_customer_classification.ipynb`)
- **Pembagian Data:** Memuat berkas `data_clustering.csv` lalu memisahkan data fitur dengan kolom `Target` ke dalam subset data pelatihan (*Train*) dan data pengujian (*Test*).
- **Baseline Modeling:** Membangun arsitektur awal klasifikasi menggunakan algoritma tunggal **Decision Tree Classifier**.
- **Ensemble Learning Exploration:** Melatih algoritma berbasis kelompok pohon keputusan acak (**Random Forest Classifier**) untuk meningkatkan stabilitas prediksi data.
- **Hyperparameter Tuning:** Melakukan pencarian kombinasi parameter terbaik menggunakan `GridSearchCV` atau `RandomizedSearchCV` (menguji variasi parameter seperti `criterion` ['gini', 'entropy'], `max_depth`, `min_samples_split`, dan `min_samples_leaf`).
- **Evaluasi Performa:** Mengukur keberhasilan model final memanfaatkan fungsi `classification_report` untuk meninjau secara mendalam nilai *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.

## Model yang Digunakan
Seluruh berkas model biner hasil pelatihan disimpan ke dalam folder `models/` menggunakan modul `joblib` agar dapat dimuat kembali secara instan untuk kebutuhan operasional (*deployment*):
- **`model_clustering.h5`**: Model latih K-Means untuk proses segmentasi perilaku objek nasabah baru.
- **`PCA_model_clustering.h5`**: Model transformator PCA untuk proyeksi fitur ke dalam koordinat komponen utama.
- **`decision_tree_model.h5`**: Model klasifikasi awal berbasis pohon keputusan tunggal.
- **`explore_random_forest_classification.h5`**: Model eksplorasi awal berbasis algoritma Random Forest.
- **`tuning_classification.h5`**: Model klasifikasi terbaik (*final optimized model*) hasil pencarian ruang parameter komprehensif.

## Hasil & Evaluasi
- **K-Means Clustering:** Segmentasi berhasil mengelompokkan 2.512 sampel nasabah secara tegas ke dalam klaster perilaku transaksi yang terpisah baik berdasarkan karakteristik keuangan (seperti kelompok nasabah dengan akumulasi saldo besar vs kelompok dengan durasi transaksi panjang dan nominal kecil).
- **Pemodelan Klasifikasi:** Penerapan algoritma *Random Forest* yang dikombinasikan dengan pencarian parameter optimal (`GridSearchCV`/`RandomizedSearchCV`) berhasil mendongkrak performa metrik ketepatan prediksi pada data pengujian (*test set*), mengurangi potensi *overfitting* secara drastis, serta menghasilkan performa klasifikasi final (`tuning_classification.h5`) yang memiliki nilai akurasi tinggi dan seimbang di seluruh label kelas target.

## Kesimpulan
Integrasi alur kerja *hybrid* (penggabungan teknik *Unsupervised* dan *Supervised Learning*) terbukti sangat tangguh untuk memproses data transaksi keuangan perbankan. Melalui teknik klastering K-Means dan reduksi dimensi PCA, karakteristik perilaku nasabah yang sebelumnya acak dan tersembunyi berhasil dipetakan secara terstruktur. Pipeline klasifikasi final yang telah dioptimalkan parameternya melengkapi proyek ini sebagai alat prediktif otomatis yang andal, memungkinkan sistem mendeteksi secara langsung kecenderungan profil kelompok dari data transaksi nasabah baru untuk mendukung pengambilan keputusan bisnis yang cerdas, efisien, dan aman.
