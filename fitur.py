"""
fitur.py
========
Pembersihan dan feature engineering. Fungsi ini dipakai di dua tempat:
1. saat pelatihan (sebagai langkah pertama pipeline),
2. saat aplikasi Streamlit membuat prediksi.

Karena fungsi ini ikut tersimpan di dalam pipeline (joblib/pickle), fungsi
harus didefinisikan di modul yang bisa di-import (bukan lambda).
"""

import numpy as np
import pandas as pd

# Kolom mentah yang dibutuhkan model (tanpa accident_id, hari, dan jumlah_korban)
KOLOM_MASUKAN = [
    "tanggal", "jam", "kota", "tipe_jalan", "kondisi_cuaca",
    "kondisi_pencahayaan", "kondisi_permukaan_jalan", "jenis_kendaraan",
    "jumlah_kendaraan_terlibat", "batas_kecepatan_kmh", "usia_pengemudi",
    "jenis_kelamin_pengemudi", "alkohol_terdeteksi",
]

# Fitur numerik dan kategorik yang masuk ke model (setelah feature engineering)
NUM = ["jam", "jumlah_kendaraan_terlibat", "batas_kecepatan_kmh", "usia_pengemudi"]
KAT = [
    "kota", "tipe_jalan", "kondisi_cuaca", "kondisi_pencahayaan",
    "kondisi_permukaan_jalan", "jenis_kendaraan", "jenis_kelamin_pengemudi",
    "alkohol_terdeteksi", "hari_dari_tanggal", "akhir_pekan", "musim",
    "kelompok_jam", "kelompok_usia",
]

KELAS = ["Ringan", "Sedang", "Berat"]

NAMA_HARI = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis", 4: "Jumat", 5: "Sabtu", 6: "Minggu"}
BULAN_MUSIM_HUJAN = [11, 12, 1, 2, 3, 4]


def fitur_turunan(df):
    """Bersihkan data mentah lalu bentuk fitur turunan. Mengembalikan DataFrame baru."""
    d = df.copy()

    # 1. Normalisasi teks: spasi, simbol '@' yang salah ketik, dan kapitalisasi
    for c in ["kota", "tipe_jalan", "kondisi_pencahayaan", "kondisi_permukaan_jalan",
              "jenis_kendaraan", "jenis_kelamin_pengemudi", "alkohol_terdeteksi"]:
        d[c] = d[c].astype("string").str.strip()
    d["kondisi_cuaca"] = (
        d["kondisi_cuaca"].astype("string").str.strip()
        .str.replace("@", "a", regex=False)     # 'Ber@w@n' -> 'Berawan'
        .str.lower().str.title()                 # 'HUJAN LEBAT' -> 'Hujan Lebat'
    )

    # 2. Fitur dari tanggal (hari dihitung ulang, karena kolom 'hari' asli tidak konsisten)
    tgl = pd.to_datetime(d["tanggal"], errors="coerce")
    d["hari_dari_tanggal"] = tgl.dt.dayofweek.map(NAMA_HARI)
    akhir = np.where(tgl.dt.dayofweek >= 5, "Akhir pekan", "Hari kerja")
    d["akhir_pekan"] = pd.Series(akhir, index=d.index).where(tgl.notna())
    musim = np.where(tgl.dt.month.isin(BULAN_MUSIM_HUJAN), "Musim hujan", "Musim kemarau")
    d["musim"] = pd.Series(musim, index=d.index).where(tgl.notna())

    # 3. Pengelompokan angka menjadi kategori yang bermakna
    d["kelompok_jam"] = pd.cut(
        d["jam"], bins=[0, 6, 11, 15, 18, 24], right=False,
        labels=["Dini hari", "Pagi", "Siang", "Sore", "Malam"],
    ).astype("object")
    d["kelompok_usia"] = pd.cut(
        d["usia_pengemudi"], bins=[0, 25, 40, 55, 200], right=False,
        labels=["<25", "25-39", "40-54", "55+"],
    ).astype("object")

    # 4. Kategori kosong dinyatakan sebagai np.nan agar SimpleImputer aman
    for c in KAT:
        d[c] = d[c].astype("object").where(d[c].notna(), np.nan)

    return d
