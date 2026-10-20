# Laporan Studi Kasus — Klasifikasi Penyebab Peringatan Deforestasi

**Metode:** Rule-Based vs. Machine Learning Konvensional pada Data Simulasi
**Sumber materi:** Webinar *"Deforestation Alerts and Driver Classification"* —
Dr. Bart Slagter (Wageningen University & Research)
**Artefak:** `notebooks/studi_kasus_deforestation_drivers.ipynb`
**Tanggal:** 20 Oktober 2026

> **Catatan:** seluruh data yang dipakai adalah **data simulasi (sintetis)** dengan
> seed tetap (42). Semua angka di bawah menggambarkan karakteristik
> **metodologi**, bukan kinerja model terhadap deforestation nyata.

---

## 1. Latar Belakang

Peringatan deforestasi (*deforestation alerts*) dihasilkan dari citra satelit
yang melintasi hutan tropis secara rutin — Sentinel-1 (radar) dan Sentinel-2
(optik) dari European Space Agency, ditambah sistem seperti GLAD, RADD, dan
TreeCoverer. Dengan pipeline pemrosesan citra, sistem ini dapat mendeteksi
kehilangan tutupan pohon dengan skala piksel 10–30 meter dan frekuensi sekali
per minggu.

Namun, alert hanya menjawab pertanyaan **"di mana"**, bukan **"mengapa"**.
Dr. Bart Slagter menjelaskan bahwa masalah "mengapa" diselesaikan dengan
**klasifikasi penyebab langsung (*direct driver*)** menggunakan *convolutional
neural network* yang membaca citra Sentinel-1 dan Sentinel-2 sebagai
*stack*. Kelas penyebab yang diklasifikasikan: pertanian skala kecil,
pertanian skala besar, penebangan selektif, pembangunan jalan, pertambangan,
kebakaran hutan, dan gangguan alami (banjir).

Studi kasus ini mengadopsi pendekatan tersebut pada level yang dapat dijalankan
mahasiswa: menguji apa yang terjadi bila masalah diselesaikan dengan pendekatan
**konvensional** — rule-based, Decision Tree, dan Random Forest — di atas fitur
tabular turunan alert.

## 2. Tujuan

1. Membangun dataset simulasi peringatan deforestasi yang realistis (1.500 baris,
   7 kelas, noise label 12%) dan reproducible.
2. Mendesain baseline rule-based yang bisa ditafsirkan manusia.
3. Membandingkan rule-based, Decision Tree, dan Random Forest (accuracy,
   precision, recall, F1, confusion matrix).
4. Menganalisis di mana model gagal dan mengaitkannya dengan temuan webinar
   (konfusi *kebakaran* vs *gangguan alami/banjir*).
5. Menguji keandalan model dalam skenario pembaruan bulanan (latensi operasional).
6. Merumuskan kapan *deep learning* perlu dipertimbangkan dan mengapa validasi
   lapangan tetap diperlukan.

## 3. Metode

**a. Desain data simulasi.** Tujuh kelas dibuat mengikuti logika
geografis-fisik yang dijelaskan pemateri:

| Kelas | Pola yang disimulasikan |
|---|---|
| `pertanian_kecil` | luas kecil, bentuk kompak, relatif dekat jalan |
| `pertanian_besar` | luas besar, bentuk persegi, frekuensi tinggi di musim kemarau |
| `selective_logging` | sedang-kecil, sangat dekat jalan akses logging, bentuk agak memanjang |
| `jalan` | luas kecil, bentuk sangat memanjang, jarak ke jalan ≈ 0 |
| `tambang` | sangat dekat konsesi tambang, bentuk kompak |
| `kebakaran` | hotspot present, luas besar, terkonsentrasi Jun–Sep, sering dekat sungai |
| `alami` | sangat dekat sungai, terkonsentrasi musim hujan, 30% punya hotspot (sumber konfusi) |

Sebesar **12% label diberi noise** ke kelas yang mirip (mis. `kebakaran` →
`alami`) untuk meniru ground truth yang ambigu; derau kecil juga ditambahkan
pada seluruh fitur numerik. Distribusi fitur memakai lognormal/gamma agar
jarak selalu positif dan ber-distribusi panjang (jarak jauh).

**b. Praproses & split.** Split 80/20 *stratified* (seed 42) menjaga proporsi
kelas di data latih dan uji.

**c. Model.**

| Model | Konfigurasi |
|---|---|
| Rule-Based | 8 aturan if–else: bentuk memanjang, kedekatan konsesi, titik api, musim, kedekatan sungai/jalan, luas area |
| Decision Tree | `max_depth=8`, `min_samples_leaf=5`, `class_weight='balanced'` |
| Random Forest | `n_estimators=400`, `min_samples_leaf=2`, `class_weight='balanced_subsample'` |
| Validasi | 5-fold *stratified* cross-validation |

**d. Evaluasi.** Accuracy, precision/recall/F1 per kelas (macro & weighted),
confusion matrix heatmap, tabel perbandingan, feature importance (gini &
permutation), visualisasi pohon (kedalaman dibatasi 4), analisis pasangan kelas
tertukar, analisis error per musim, serta backtest pembaruan bulanan
(latih Jan–Sep, uji Okt–Des).

## 4. Hasil

**Karakteristik data.** 1.500 alert, 7 kelas, 1.200 data latih / 300 data uji.
Distribusi kelas cukup seimbang (10,6%–21,9% per kelas), sehingga metrik
*macro-averaged* lebih bermakna daripada *accuracy* saja.

**Perbandingan model (data uji, n = 300).**

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | 5-Fold CV |
|---|---|---|---|---|---|---|
| Rule-Based (if–else) | 0,7067 | 0,7382 | 0,7235 | 0,7134 | 0,6986 | — |
| Decision Tree | 0,7533 | 0,7513 | 0,7652 | 0,7548 | 0,7519 | 0,7633 ± 0,0236 |
| **Random Forest** | **0,8200** | **0,8200** | **0,8276** | **0,8193** | **0,8178** | 0,8167 ± 0,0257 |

**Metrik per kelas — Random Forest.**

| Kelas | Precision | Recall | F1 | Error rate |
|---|---|---|---|---|
| `jalan` | 0,762 | **1,000** | 0,865 | **0,000** |
| `pertanian_besar` | 0,833 | 0,897 | 0,864 | 0,103 |
| `tambang` | 0,919 | 0,895 | 0,907 | 0,105 |
| `selective_logging` | 0,870 | 0,825 | 0,847 | 0,175 |
| `kebakaran` | 0,750 | 0,794 | 0,771 | 0,206 |
| `pertanian_kecil` | 0,790 | 0,754 | 0,772 | 0,246 |
| `alami` | 0,815 | **0,629** | 0,710 | **0,371** |

**Konfusi `kebakaran` vs `alami` (banjir).** 2 alert kebakaran salah dikira
alami (5,9%), 5 alert alami salah dikira kebakaran (14,3%); total 7 alert
(2,3% dari data uji).

**Pasangan tertukar terbesar (RF).** `alami` → `pertanian_kecil` (7),
`pertanian_kecil` → `jalan` (5), `alami` → `kebakaran` (5),
`selective_logging` → `pertanian_kecil` (4), `kebakaran` → `pertanian_besar` (4).

**Pengaruh musim.** Recall `alami`: 0,690 (musim hujan) → 0,333 (musim kemarau);
recall `kebakaran` stabil (0,789–0,800).

**Feature importance (permutation).** `kedekatan_tambang_km` (0,2467) >
`luas_ha` (0,2237) > `jarak_ke_jalan_km` (0,1717) > `jarak_ke_sungai_km` (0,0803) >
`bentuk_memanjang` (0,0650) > `ada_titik_api` (0,0497) > `bulan` (0,0060).

**Backtest bulanan.** Okt 0,803, Nov 0,827, Des 0,832; rata-rata **0,8207**
(≈ split acak 0,8200).

## 5. Pembahasan

**a. Mengapa Random Forest menang?** Forest mengagregasi ratusan pohon dengan
*bootstrap sampling* dan *feature randomness*, sehingga lebih tahan overfit dan
lebih stabil. Pohon juga menangkap interaksi non-linier antar fitur
("dekat sungai **dan** ada hotspot **dan** bulan kemarau") yang mustahil dituliskan
dengan if–else sederhana, serta belajar dari *konteks* — sesuai penjelasan
pemateri.

**b. Mengapa rule-based kalah jauh?** Aturan manual hanya memanfaatkan satu-satu fitur
secara berurutan. **Urutan aturan** menentukan hasil: pada seluruh data, 112 alert
`selective_logging` salah diklasifikasikan sebagai `pertanian_kecil`; pada data
uji, 20 dari 57 alert (35%) mengalami hal serupa, sehingga recall kelas ini
hanya 0,439 dengan precision 0,893 — model sangat konservatif dan banyak alert
logging tidak pernah dikenali. Aturan bisa diperbaiki manual, tetapi setiap
perbaikan harus diuji ulang ke seluruh data.

**c. Mengapa masih ada error ±12%?** Karena error itu ditanamkan pada label
secara sengaja. Batas atas teoritis akurasi model pada data ini ≈ 0,88, dan
Random Forest sudah mencapainya (0,8200). Kesimpulan metodologisnya: **kualitas
label lebih menentukan daripada arsitektur model**.

**d. Relevansi dengan materi.** Data simulasi meniru temuan pemateri —
kebakaran muncul di sepanjang sungai, dan banjir terjadi di lokasi yang sama.
Pada Q&A, pemateri menambahkan bahwa Sentinel-1 sangat sensitif terhadap air
tetapi penetrasinya ke kanopi rapat terbatas. Untuk membedakan *fire* dan
*flood*, kombinasi citra optik multi-temporal (Sentinel-2) + informasi musiman
jauh lebih informatif daripada satu fitur biner `ada_titik_api`. Hasil eksperimen
mendukungnya secara kuantitatif.

**e. Kapan deep learning perlu dipertimbangkan?** Bila: (1) input berupa
*stack* citra multi-sensor, (2) pola bersifat spasial-temporal (mis. rangkaian
*mining → agriculture*), (3) data berlabel banyak, atau (4) konteks tidak dapat
dirumuskan sebagai fitur tabel. Kesimpulannya bukan "deep learning tidak
berguna", melainkan "deep learning harus dibayar dengan data berlabel mahal".

**f. Dampak operasional.** Presisi dan recall sama-sama penting: presisi
rendah berarti *false alarm*, recall rendah berarti deforestasi ilegal yang lolos
radar. Klasifikasi penyebab membantu memprioritaskan alert; kelas `alami` yang
paling lemah recall-nya adalah kandidat prioritas perbaikan, karena alert banjir
yang salah dikira kebakaran justru berisiko memicu tindakan hukum yang tidak
berdasar.

**g. Keterbatasan data simulasi.** Ground truth sintetis terlalu bersih; hanya
7 fitur tabuler (tanpa citra Sentinel-1/2); tidak ada variabel geografis
(wilayah, negara, vegetasi, kepemilikan lahan) padahal pemateri menunjukkan
perbedaan kuat antar wilayah; musim memakai pola global sederhana; tidak ada
*concept drift* maupun kegagalan satelit; biaya anotasi label tinggi tidak
tergantikan. **Tanpa validasi lapangan, tidak ada angka di laporan ini yang
boleh dipakai sebagai estimasi kinerja di dunia nyata.**

## 6. Kesimpulan

1. **Rule-based berguna sebagai baseline pertama** (accuracy 0,7067) tetapi gagal
   pada kasus tumpang-tindih: recall `selective_logging` 0,439 dan `alami` 0,543.
2. **Random Forest terbaik** (accuracy 0,8200; macro-F1 0,8193; 5-fold CV
   0,8167 ± 0,0257), konsisten mengungguli Decision Tree (0,7533) dan
   rule-based (0,7067).
3. **Tidak ada model yang sempurna**, dan itu seharusnya: label penyebab
   deforestasi ambigu di lapangan. **Kualitas label adalah pembatas utama.**
4. **Error terkonsentrasi pada pasangan ambigu secara fisis**:
   *kebakaran* ↔ *alami/banjir* (7 dari 300 alert uji), dengan `alami` sebagai
   kelas terlemah dan recall yang anjlok 0,690 → 0,333 antara musim hujan dan
   kemarau. Perbaikan perlu data tambahan (citra optik multi-temporal, curah
   hujan, hotspot historis, validasi lapangan), bukan sekadar model lebih kompleks.
5. **Model stabil terhadap pergeseran musiman pada data simulasi** (backtest
   0,8207 vs 0,8200), namun sistem peringatan bulanan tetap perlu monitoring
   dan retraining berkala.
6. **Deep learning layak dipertimbangkan** pada tahap berikutnya — dengan data
   berlabel jauh lebih banyak, input Sentinel-1 + Sentinel-2, dan kebutuhan
   membaca konteks spasial-temporal — tetapi hasilnya **wajib divalidasi dengan
   data lapangan**, karena citra satelit hanya menunjukkan *jejak* dan *konteks*,
   bukan sebab yang sebenarnya.

## 7. Lampiran

| Berkas | Isi |
|---|---|
| `data/alert_deforestation_simulasi.csv` | 1.500 alert sintetis (seed 42) |
| `data/perbandingan_model.csv` | metrik tiga model |
| `data/analisis_kesalahan_per_kelas.csv` | error rate per kelas |
| `data/pasangan_kelas_tertukar.csv` | 10 pasangan tertukar terbesar |
| `data/figures/*.png` | 13 grafik (distribusi, seasonality, korelasi, CV, confusion matrix, feature importance, pohon, backtest) |
