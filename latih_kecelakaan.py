"""
latih_kecelakaan.py
===================
Melatih model klasifikasi tingkat_keparahan (Ringan / Sedang / Berat),
mengevaluasinya, dan menyimpan pipeline ke models/ dengan joblib.
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fitur import KAT, KELAS, KOLOM_MASUKAN, NUM, fitur_turunan

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# --- 1. Muat data dan buang duplikat ---
df = pd.read_csv("data/traffic_accident.csv")
df = df.drop_duplicates(subset="accident_id").reset_index(drop=True)
df = df.dropna(subset=["tingkat_keparahan"])
print("Data setelah dedup:", df.shape)

X = df[KOLOM_MASUKAN]          # jumlah_korban, accident_id, dan hari TIDAK dipakai
y = df["tingkat_keparahan"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
)

# --- 2. Preprocessing: imputasi, scaling, dan one-hot ---
prep = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUM),
    ("kat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), KAT),
])

# --- 3. Bandingkan dua model dengan cross-validation di data TRAIN ---
kandidat = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1),
}

skor_cv = {}
for nama, clf in kandidat.items():
    pipe = Pipeline([
        ("fe", FunctionTransformer(fitur_turunan)),
        ("prep", prep),
        ("clf", clf),
    ])
    skor = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro")
    skor_cv[nama] = skor.mean()
    print(f"{nama:20} F1-macro CV = {skor.mean():.3f} (± {skor.std():.3f})")

nama_terpilih = max(skor_cv, key=skor_cv.get)
print("Model terpilih:", nama_terpilih)

# --- 3b. Baseline: tebakan acak sesuai proporsi kelas (pembanding minimum) ---
baseline = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
baseline_pipe = Pipeline([("fe", FunctionTransformer(fitur_turunan)), ("prep", prep), ("clf", baseline)])
baseline_pipe.fit(X_train, y_train)
print(f"Baseline (tebakan acak) F1-macro TEST = "
      f"{f1_score(y_test, baseline_pipe.predict(X_test), average='macro'):.3f}")

# --- 4. Latih model terpilih pada seluruh data train, evaluasi sekali di TEST ---
model = Pipeline([
    ("fe", FunctionTransformer(fitur_turunan)),
    ("prep", prep),
    ("clf", kandidat[nama_terpilih]),
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)
kelas_model = list(model.classes_)

metrik = {
    "accuracy": accuracy_score(y_test, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "f1_macro": f1_score(y_test, y_pred, average="macro"),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    "roc_auc_ovr_macro": roc_auc_score(
        y_test, proba, multi_class="ovr", average="macro", labels=kelas_model),
}
print("\n=== Metrik pada data TEST ===")
for k, v in metrik.items():
    print(f"{k:20} {v:.3f}")

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, labels=KELAS, digits=3))

cm = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=KELAS),
    index=[f"Aktual {k}" for k in KELAS],
    columns=[f"Prediksi {k}" for k in KELAS],
)
print("=== Confusion matrix ===")
print(cm)

# --- 5. Simpan bundle dengan joblib ---
Path("models").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
cm.to_csv("output/confusion_matrix.csv")

fe_df = fitur_turunan(X)
bundle = {
    "pipeline": model,
    "kolom_masukan": KOLOM_MASUKAN,
    "kelas": kelas_model,
    "opsi": {c: sorted(fe_df[c].dropna().unique().tolist())
             for c in ["kota", "tipe_jalan", "kondisi_cuaca", "kondisi_pencahayaan",
                       "kondisi_permukaan_jalan", "jenis_kendaraan",
                       "jenis_kelamin_pengemudi", "alkohol_terdeteksi"]},
    "metrik_test": {k: float(v) for k, v in metrik.items()},
    "confusion_matrix": cm,
    "model_terpilih": nama_terpilih,
    "versi_sklearn": sklearn.__version__,
}
joblib.dump(bundle, "models/keparahan_kecelakaan.joblib", compress=3)
print("\nModel tersimpan di models/keparahan_kecelakaan.joblib")
