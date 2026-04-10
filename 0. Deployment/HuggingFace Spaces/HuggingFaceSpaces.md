# Deploy ke HuggingFace Spaces dengan Streamlit

## Daftar Isi

1. [Apa itu HuggingFace Spaces?](#apa-itu-huggingface-spaces)
2. [Persiapan Lokal](#persiapan-lokal)
3. [Struktur Project](#struktur-project)
4. [Membuat Streamlit App](#membuat-streamlit-app)
5. [Upload ke HF Spaces](#upload-ke-hf-spaces)
6. [Tips dan Troubleshooting](#tips-dan-troubleshooting)

---

## Apa itu HuggingFace Spaces?

**HuggingFace Spaces** adalah platform hosting gratis dari HuggingFace untuk men-deploy aplikasi ML secara publik. Mendukung Streamlit, Gradio, dan Docker — tanpa perlu setup server sendiri.

> **Analogi:** Spaces itu seperti Google Drive, tapi khusus untuk aplikasi ML. Kamu upload kodenya, HuggingFace yang jalankan servernya, dan semua orang bisa akses lewat link publik. Gratis pula! 🤗

**Keunggulan:**
- Gratis (dengan batasan resource)
- Link publik otomatis (`username.hf.space/space-name`)
- Terintegrasi dengan HuggingFace Hub (model, dataset)
- Support Git untuk update

---

## Persiapan Lokal

### 1. Install dependencies

```bash
pip install streamlit scikit-learn joblib numpy pandas
```

### 2. Latih dan simpan model

```python
# train_and_save.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

# Simpan pipeline
joblib.dump(pipeline, 'model.joblib')
print("Model disimpan ke model.joblib")
print(f"Akurasi test: {pipeline.score(X_test, y_test):.4f}")
```

### 3. Test aplikasi lokal

```bash
streamlit run app.py
```

---

## Struktur Project

Struktur folder yang dibutuhkan HuggingFace Spaces:

```
my-ml-app/
├── app.py              ← Entry point utama (wajib bernama app.py)
├── model.joblib        ← Model yang sudah dilatih
├── requirements.txt    ← Daftar dependencies
└── README.md           ← Deskripsi Space (opsional tapi dianjurkan)
```

> ⚠️ **Penting:** HuggingFace Spaces dengan SDK Streamlit **wajib** punya file bernama `app.py` sebagai entry point.

---

## Membuat Streamlit App

Berikut contoh aplikasi lengkap untuk klasifikasi Iris:

```python
# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---- Konfigurasi halaman ----
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

# ---- Load model (cache agar tidak reload tiap interaksi) ----
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

pipeline    = load_model()
CLASS_NAMES = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
CLASS_EMOJI = ['🌸', '🌺', '🌼']

# ---- Header ----
st.title("🌸 Iris Flower Classifier")
st.write("Prediksi jenis bunga Iris berdasarkan ukuran kelopak dan mahkotanya.")
st.divider()

# ---- Layout dua kolom ----
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Input Pengukuran")

    sepal_length = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width  = st.slider("🌿 Sepal Width (cm)",  2.0, 4.5, 3.5, 0.1)
    petal_length = st.slider("🌺 Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width  = st.slider("🌺 Petal Width (cm)",  0.1, 2.5, 0.2, 0.1)

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    predict_btn = st.button("Prediksi Sekarang", type="primary", use_container_width=True)

with col_result:
    st.subheader("Hasil Prediksi")

    if predict_btn:
        prediction  = pipeline.predict(features)[0]
        probability = pipeline.predict_proba(features)[0]

        # Tampilkan prediksi utama
        st.success(f"{CLASS_EMOJI[prediction]} **{CLASS_NAMES[prediction]}**")
        st.metric(
            label="Confidence",
            value=f"{probability[prediction]:.1%}"
        )

        st.write("**Probabilitas semua kelas:**")
        prob_df = pd.DataFrame({
            'Kelas': [f"{e} {n}" for e, n in zip(CLASS_EMOJI, CLASS_NAMES)],
            'Probabilitas': probability
        })
        st.bar_chart(prob_df.set_index('Kelas'))

    else:
        st.info("Atur slider di sebelah kiri lalu klik **Prediksi Sekarang**.")

# ---- Info tambahan ----
st.divider()
with st.expander("Tentang Model"):
    st.write("""
    Model ini menggunakan **Random Forest Classifier** yang dilatih pada dataset Iris klasik.
    Dataset terdiri dari 150 sampel, 4 fitur, dan 3 kelas bunga.
    """)
    st.write(f"Jumlah fitur input : **{pipeline.n_features_in_}**")
    st.write(f"Kelas yang diprediksi : **{', '.join(CLASS_NAMES)}**")
```

---

## Upload ke HF Spaces

### Cara 1: Lewat Website (Termudah)

1. Buka [huggingface.co](https://huggingface.co) dan login / buat akun
2. Klik **New Space**
3. Isi nama Space, pilih **Streamlit** sebagai SDK
4. Pilih visibility: **Public** (gratis) atau **Private**
5. Klik **Create Space**
6. Upload file satu per satu lewat tab **Files** di Space kamu:
   - `app.py`
   - `model.joblib`
   - `requirements.txt`

### Cara 2: Lewat Git (Direkomendasikan)

```bash
# 1. Install git-lfs untuk file besar (model)
git lfs install

# 2. Clone repo Space yang baru dibuat
git clone https://huggingface.co/spaces/USERNAME/SPACE-NAME
cd SPACE-NAME

# 3. Salin semua file project ke sini
cp /path/to/your/project/app.py .
cp /path/to/your/project/model.joblib .
cp /path/to/your/project/requirements.txt .

# 4. Track file besar dengan git-lfs
git lfs track "*.joblib"
git lfs track "*.pkl"
git add .gitattributes

# 5. Commit dan push
git add .
git commit -m "Initial deployment"
git push
```

Space akan otomatis build dan berjalan dalam beberapa menit.

### File `requirements.txt`

```txt
streamlit==1.35.0
scikit-learn==1.5.0
joblib==1.4.2
numpy==1.26.4
pandas==2.2.2
```

> ✅ **Selalu pin versi** — tanpa pin versi, update library bisa merusak model yang sudah disimpan.

### File `README.md` (untuk metadata Space)

```markdown
---
title: Iris Flower Classifier
emoji: 🌸
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Iris Flower Classifier

Aplikasi klasifikasi bunga Iris menggunakan Random Forest.
```

---

## Tips dan Troubleshooting

### Model terlalu besar (> 10 MB)

Gunakan **git-lfs** (wajib untuk file > 10 MB):

```bash
git lfs install
git lfs track "*.joblib"
git lfs track "*.pkl"
git lfs track "*.onnx"
git add .gitattributes
git add model.joblib
git commit -m "add model with lfs"
git push
```

### Space lambat saat pertama dibuka

Spaces gratis menggunakan CPU dan "tidur" setelah tidak aktif. Wajar jika loading pertama memakan 30-60 detik. Gunakan `@st.cache_resource` agar model hanya load sekali.

### Error `ModuleNotFoundError`

Pastikan semua library ada di `requirements.txt` dengan versi yang tepat:

```bash
# Generate requirements.txt otomatis dari environment
pip freeze > requirements.txt

# Atau hanya package yang dibutuhkan
pip list --format=freeze | grep -E "streamlit|scikit|joblib|numpy|pandas" > requirements.txt
```

### Model tidak kompatibel setelah update scikit-learn

```python
# Cek versi scikit-learn saat training
import sklearn
print(sklearn.__version__)

# Simpan versi ke dalam artifact
import joblib
joblib.dump({
    'model': pipeline,
    'sklearn_version': sklearn.__version__
}, 'artifacts.joblib')
```

### Perbandingan Resource HuggingFace Spaces

| Tier | CPU | RAM | Storage | Harga |
|------|-----|-----|---------|-------|
| **Free** | 2 vCPU | 16 GB | 50 GB | Gratis |
| **Persistent Storage** | 2 vCPU | 16 GB | 50 GB + storage | $5/bln |
| **T4 Small (GPU)** | 4 vCPU | 15 GB | 50 GB | $0.60/jam |
| **A10G Small (GPU)** | 4 vCPU | 15 GB | 50 GB | $1.05/jam |

---

## Referensi

- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [HuggingFace git-lfs Guide](https://huggingface.co/docs/hub/repositories-getting-started#terminal)
