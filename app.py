"""
app.py
======
Aplikasi Streamlit untuk memprediksi tingkat keparahan kecelakaan.
Jalankan: streamlit run app.py
"""

import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "keparahan_kecelakaan.joblib"


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)   # memuat pipeline: feature engineering + model


st.set_page_config(page_title="Prediksi Keparahan Kecelakaan", layout="wide")
st.title("Prediksi Tingkat Keparahan Kecelakaan")
st.caption("Model klasifikasi: Ringan, Sedang, atau Berat. Dipakai sebagai alat bantu, bukan keputusan akhir.")

bundle = load_bundle()
if bundle is None:
    st.error("Model belum ada. Jalankan latih_kecelakaan.py terlebih dahulu.")
    st.stop()

opsi = bundle["opsi"]
tab_prediksi, tab_metrik = st.tabs(["Prediksi", "Metrik evaluasi"])

with tab_prediksi:
    kiri, kanan = st.columns(2)
    with kiri:
        tanggal = st.date_input("Tanggal kejadian", value=dt.date(2024, 6, 1),
                                min_value=dt.date(2020, 1, 1), max_value=dt.date(2024, 12, 31))
        jam = st.slider("Jam kejadian", 0, 23, 17)
        kota = st.selectbox("Kota", opsi["kota"])
        tipe_jalan = st.selectbox("Tipe jalan", opsi["tipe_jalan"])
        cuaca = st.selectbox("Kondisi cuaca", opsi["kondisi_cuaca"])
        pencahayaan = st.selectbox("Kondisi pencahayaan", opsi["kondisi_pencahayaan"])
    with kanan:
        permukaan = st.selectbox("Kondisi permukaan jalan", opsi["kondisi_permukaan_jalan"])
        kendaraan = st.selectbox("Jenis kendaraan", opsi["jenis_kendaraan"])
        jumlah_kendaraan = st.number_input("Jumlah kendaraan terlibat", 1, 5, 2)
        batas = st.number_input("Batas kecepatan (km/jam)", 40, 100, 60, step=10)
        usia = st.number_input("Usia pengemudi", 17, 72, 35)
        jk = st.selectbox("Jenis kelamin pengemudi", opsi["jenis_kelamin_pengemudi"])
        alkohol = st.selectbox("Alkohol terdeteksi", opsi["alkohol_terdeteksi"])

    if st.button("Prediksi", type="primary"):
        masukan = pd.DataFrame([{
            "tanggal": str(tanggal),
            "jam": float(jam),
            "kota": kota,
            "tipe_jalan": tipe_jalan,
            "kondisi_cuaca": cuaca,
            "kondisi_pencahayaan": pencahayaan,
            "kondisi_permukaan_jalan": permukaan,
            "jenis_kendaraan": kendaraan,
            "jumlah_kendaraan_terlibat": float(jumlah_kendaraan),
            "batas_kecepatan_kmh": float(batas),
            "usia_pengemudi": float(usia),
            "jenis_kelamin_pengemudi": jk,
            "alkohol_terdeteksi": alkohol,
        }])[bundle["kolom_masukan"]]

        proba = bundle["pipeline"].predict_proba(masukan)[0]
        hasil = pd.Series(proba, index=bundle["kelas"], name="probabilitas")
        pred = hasil.idxmax()

        st.subheader(f"Prediksi: {pred}")
        st.bar_chart(hasil)
        st.dataframe(hasil.map(lambda p: f"{p:.1%}").to_frame())

with tab_metrik:
    st.subheader(f"Model terpilih: {bundle['model_terpilih']}")
    st.markdown("**Metrik pada data test (600 kejadian yang tidak dipakai melatih model):**")
    m = bundle["metrik_test"]
    kolom = st.columns(len(m))
    for col, (nama, nilai) in zip(kolom, m.items()):
        col.metric(nama, f"{nilai:.3f}")
    st.markdown("**Confusion matrix:**")
    st.dataframe(bundle["confusion_matrix"])
