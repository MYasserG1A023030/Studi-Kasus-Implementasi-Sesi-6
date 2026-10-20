# Studi Kasus Implementasi Sesi 6

Studi kasus machine learning geospasial yang disusun dari materi webinar
*"Deforestation Alerts and Driver Classification"* (Program Studi Informatika;
narasumber Dr. Bart Slagter, Wageningen University & Research). Repo ini
berisi satu notebook Jupyter yang dapat dijalankan ulang, kerangka laporan siap
isi, dan seluruh keluaran analisis.

| Isi repo | Keterangan |
| --- | --- |
| `notebooks/` | Notebook utama (43 sel): data simulasi, EDA, rule-based, Decision Tree, Random Forest, evaluasi, analisis kesalahan, kerangka laporan |
| `docs/` | Laporan studi kasus beserta temuan, pembahasan, dan kesimpulan |
| `data/` | CSV data simulasi, 3 tabel ringkasan hasil, dan 13 grafik keluaran |

## Studi Kasus Klasifikasi Penyebab Peringatan Deforestasi

Peringatan deforestasi dari citra Sentinel-1 dan Sentinel-2 hanya menunjukkan
**di mana** tutupan pohon hilang, bukan **mengapa**. Studi kasus ini
menggambarkan alur klasifikasi penyebab langsung (*direct driver*) — antara
lain pertanian kecil/besar, penebangan selektif, jalan, tambang, kebakaran, dan
gangguan alami — dengan pendekatan konvensional (rule-based, Decision Tree,
Random Forest), lalu mengaitkan hasilnya dengan temuan pemateri.

| Aspek | Nilai |
| --- | --- |
| Objek | Peringatan deforestasi (alert) pada hutan tropis, skala piksel 10–30 m |
| Target | 7 penyebab langsung: `pertanian_kecil`, `pertanian_besar`, `selective_logging`, `jalan`, `tambang`, `kebakaran`, `alami` |
| Data | 1.500 baris x 8 kolom (7 fitur + label), simulasi dengan seed 42 |
| Sasaran | Menentukan penyebab alert secara otomatis dan menguji keandalan tiap pendekatan |
| Tool | Python 3.11 (pandas, numpy, matplotlib, seaborn, scikit-learn) |

## Temuan utama (dihitung otomatis oleh notebook)

- **Random Forest terbaik** di antara tiga pendekatan: akurasi **0,8200**,
  macro-F1 **0,8193**, 5-fold CV **0,8167 ± 0,0257** — konsisten mengungguli
  Decision Tree (0,7533) dan rule-based (0,7067).
- Tidak ada model yang mencapai akurasi sempurna karena **12% label diberi
  noise** secara disengaja, sehingga batas atas teoretis akurasi model pada data
  ini ≈ 0,88. Kesimpulannya: **kualitas label lebih menentukan daripada
  arsitektur model**.
- **Kelas `alami` (banjir) paling lemah** (recall 0,629; error rate 0,371),
  sedangkan `jalan` sempurna (recall 1,000). Kesalahan terkonsentrasi pada
  pasangan yang memang ambigu secara fisis: *kebakaran* ↔ *alami* (7 dari 300
  alert uji).
- **Musim menentukan**: recall `alami` anjlok dari **0,690** (musim hujan)
  menjadi **0,333** (musim kemarau) karena sinyal hotspot ikut muncul.
- Rule-based gagal pada kasus tumpang-tindih: recall `selective_logging` hanya
  **0,439** karena urutan aturan `tambang` diperiksa lebih dulu (112 alert
  logging salah diklasifikasikan pada seluruh data).
- **Feature importance** menyiratkan model membaca konteks, bukan hanya bentuk:
  `kedekatan_tambang_km` (0,2467) > `luas_ha` (0,2237) > `jarak_ke_jalan_km`
  (0,1717) — selaras dengan penjelasan pemateri bahwa modelnya membaca
  lingkungan sekitar alert, bukan sekadar ukuran/bentuk.
- **Backtest bulanan** (latih Jan–Sep, uji Okt–Des) menghasilkan akurasi
  **0,8207**, hampir sama dengan split acak (0,8200) — stabil pada pola
  musiman, dengan asumsi tidak adanya *concept drift* dan kegagalan satelit.

## Cara menjalankan

```bash
cd Studi-Kasus-Implementasi-Sesi-6
python -m pip install -r requirements.txt
jupyter notebook notebooks/studi_kasus_deforestation_drivers.ipynb
```

Di Google Colab: `File → Import notebook` → pilih
`notebooks/studi_kasus_deforestation_drivers.ipynb` → `Runtime → Run all`.
Notebook berjalan sekitar 30 detik dan menulis keluarannya ke folder `data/`
(`/content/data/` di Colab). Seluruh proses acak memakai seed 42, sehingga
hasilnya identik setiap kali dijalankan.

## Struktur folder

```
Studi-Kasus-Implementasi-Sesi-6/
├── requirements.txt
├── README.md
├── notebooks/
│   └── studi_kasus_deforestation_drivers.ipynb  # notebook utama (siap Colab)
├── data/
│   ├── alert_deforestation_simulasi.csv         # 1.500 alert sintetis (seed 42)
│   ├── perbandingan_model.csv                   # metrik tiga model
│   ├── analisis_kesalahan_per_kelas.csv         # error rate per kelas
│   ├── pasangan_kelas_tertukar.csv              # 10 pasangan tertukar terbesar
│   └── figures/                                 # 13 grafik
└── docs/
    └── 01-laporan-studi-kasus.md                 # laporan studi kasus
```

## Dokumentasi

| Dokumen | Isi |
| --- | --- |
| [01-laporan-studi-kasus.md](docs/01-laporan-studi-kasus.md) | Latar belakang, tujuan, metode, hasil, pembahasan, dan kesimpulan studi kasus |
| [notebooks/studi_kasus_deforestation_drivers.ipynb](notebooks/studi_kasus_deforestation_drivers.ipynb) | Notebook lengkap: data, EDA, model, evaluasi, analisis kesalahan, dan kerangka laporan |

## Catatan integritas data

Seluruh data dalam studi kasus ini adalah **data sintetis** yang dibuat dengan
seed tetap (42), sehingga hasil selalu reproducible. Tidak ada data pengamatan
nyata dari Sentinel-1, Sentinel-2, GLAD, RADD, MapBiomas, atau sistem
peringatan deforestasi mana pun. Karena itu, angka pada laporan ini
menggambarkan **karakteristik metodologi dan kesalahan model**, bukan estimasi
kinerja model di dunia nyata; hasil pada data lapangan tetap memerlukan
validasi lapangan sebelum dipakai untuk tindakan penegakan hukum.
