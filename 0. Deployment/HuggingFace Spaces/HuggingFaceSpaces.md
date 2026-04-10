# Deploy ke HuggingFace Spaces dengan Docker + Streamlit

## Daftar Isi

1. [Apa itu HuggingFace Spaces?](#apa-itu-huggingface-spaces)
2. [Persiapan Lokal](#persiapan-lokal)
3. [Struktur Project](#struktur-project)
4. [Membuat Streamlit App](#membuat-streamlit-app) *(House Price Predictor)*
5. [Dockerfile](#dockerfile)
6. [Upload ke HF Spaces](#upload-ke-hf-spaces)
7. [Tips dan Troubleshooting](#tips-dan-troubleshooting)

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

Model sudah dilatih di notebook (`1. Machine Learning/2. Supervised Learning/Implementasi/notebook.ipynb`) dan menghasilkan dua file `.pkl`:

```python
# Cuplikan dari notebook — sudah dijalankan, model sudah tersimpan
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib

# ... (lihat notebook lengkapnya)

joblib.dump(lr_model, 'model_linear_regression.pkl')
joblib.dump(dt_model, 'model_decision_tree.pkl')
```

Salin kedua file model ke folder project deployment:

```bash
cp model_linear_regression.pkl my-ml-app/
cp model_decision_tree.pkl     my-ml-app/
```

### 3. Test aplikasi lokal

```bash
# Jalankan langsung (dari root folder project)
streamlit run src/streamlit_app.py

# Atau via Docker (sama persis seperti di HF Spaces)
docker build -t house-price-app .
docker run -p 7860:7860 house-price-app
# Buka http://localhost:7860
```

---

## Struktur Project

Saat membuat Space baru dengan SDK **Docker**, HuggingFace otomatis men-generate struktur berikut (dari template `streamlit/streamlit-template-space`):

```
my-ml-app/
├── src/
│   └── streamlit_app.py          ← Aplikasi Streamlit (di dalam folder src/)
├── .gitattributes                ← Konfigurasi git-lfs (sudah ada otomatis)
├── Dockerfile                    ← Instruksi build container
├── model_linear_regression.pkl   ← Model Linear Regression  ← kita tambahkan
├── model_decision_tree.pkl       ← Model Decision Tree Regressor  ← kita tambahkan
├── requirements.txt              ← Daftar dependencies Python
└── README.md                     ← Metadata Space (header YAML wajib ada)
```

> ⚠️ **Penting:** File app ada di `src/streamlit_app.py`, bukan `app.py` di root. Sesuaikan `CMD` di Dockerfile agar menunjuk ke path yang benar. Port **wajib 7860**.

---

## Membuat Streamlit App

Berikut contoh aplikasi lengkap untuk prediksi harga rumah menggunakan dua model:

```python
# src/streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---- Konfigurasi halaman ----
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---- Load model (cache agar tidak reload tiap interaksi) ----
@st.cache_resource
def load_models():
    # model .pkl ada di root, src/ ada satu level di bawah
    lr = joblib.load('model_linear_regression.pkl')
    dt = joblib.load('model_decision_tree.pkl')
    return lr, dt

lr_model, dt_model = load_models()

FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    'floors', 'waterfront', 'view', 'condition',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'
]

# ---- Header ----
st.title("🏠 House Price Predictor")
st.write("Prediksi harga rumah di King County, Washington menggunakan Linear Regression dan Decision Tree.")
st.divider()

# ---- Layout dua kolom ----
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Spesifikasi Rumah")

    bedrooms    = st.number_input("Jumlah Kamar Tidur",     min_value=1,    max_value=10,    value=3,     step=1)
    bathrooms   = st.number_input("Jumlah Kamar Mandi",     min_value=1.0,  max_value=8.0,   value=2.0,   step=0.25)
    sqft_living = st.number_input("Luas Rumah (sqft)",      min_value=300,  max_value=10000, value=1800,  step=50)
    sqft_lot    = st.number_input("Luas Tanah (sqft)",      min_value=500,  max_value=100000,value=7000,  step=500)
    floors      = st.number_input("Jumlah Lantai",          min_value=1.0,  max_value=3.5,   value=1.0,   step=0.5)
    waterfront  = st.selectbox("Tepi Pantai (Waterfront)?", options=[0, 1], format_func=lambda x: "Ya" if x else "Tidak")
    view        = st.slider("Skor View (0–4)",              min_value=0, max_value=4, value=0)
    condition   = st.slider("Kondisi Rumah (1–5)",          min_value=1, max_value=5, value=3)
    sqft_above  = st.number_input("Luas di Atas Tanah (sqft)", min_value=300, max_value=10000, value=1800, step=50)
    sqft_basement = st.number_input("Luas Basement (sqft)", min_value=0,   max_value=5000,  value=0,     step=50)
    yr_built    = st.number_input("Tahun Dibangun",         min_value=1900, max_value=2015,  value=1990,  step=1)
    yr_renovated = st.number_input("Tahun Renovasi (0 = belum)", min_value=0, max_value=2015, value=0,   step=1)

    features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot,
                          floors, waterfront, view, condition,
                          sqft_above, sqft_basement, yr_built, yr_renovated]])

    predict_btn = st.button("Prediksi Harga", type="primary", use_container_width=True)

with col_result:
    st.subheader("Hasil Prediksi")

    if predict_btn:
        pred_lr = lr_model.predict(features)[0]
        pred_dt = dt_model.predict(features)[0]

        st.metric(label="Linear Regression", value=f"${pred_lr:,.0f}")
        st.metric(label="Decision Tree",      value=f"${pred_dt:,.0f}")

        # Visualisasi perbandingan
        result_df = pd.DataFrame({
            'Model': ['Linear Regression', 'Decision Tree'],
            'Prediksi Harga ($)': [pred_lr, pred_dt]
        }).set_index('Model')
        st.bar_chart(result_df)

    else:
        st.info("Isi spesifikasi rumah di sebelah kiri lalu klik **Prediksi Harga**.")

# ---- Info tambahan ----
st.divider()
with st.expander("Tentang Model"):
    st.write("""
    Kedua model dilatih pada dataset **King County House Sales** (Washington, USA).
    - **Linear Regression** — model baseline yang mudah diinterpretasi
    - **Decision Tree Regressor** — `max_depth=6`, `min_samples_leaf=10`

    **Fitur yang digunakan:** bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated.
    """)
    st.write(f"Jumlah fitur input : **{len(FEATURES)}**")
```

---

## Dockerfile

HuggingFace meng-generate `Dockerfile` default saat Space dibuat. Kita perlu menyesuaikannya agar menunjuk ke `src/streamlit_app.py` dan menginstall dependencies kita:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file ke container
COPY . .

# HuggingFace Spaces Docker wajib expose port 7860
EXPOSE 7860

# Jalankan Streamlit — file ada di src/streamlit_app.py
CMD ["streamlit", "run", "src/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

> **Kenapa `--server.headless=true`?** Agar Streamlit tidak mencoba membuka browser saat berjalan di dalam container.

---

## Upload ke HF Spaces

### Alur: Buat Space → Clone → Edit → Push

#### 1. Buat Space baru di HuggingFace

1. Buka [huggingface.co](https://huggingface.co) dan login / buat akun
2. Klik **New Space**
3. Isi nama Space, pilih **Docker** sebagai SDK
4. Pilih visibility: **Public** (gratis) atau **Private**
5. Klik **Create Space** — HuggingFace akan generate struktur awal otomatis

#### 2. Clone repo Space ke lokal

```bash
# Install git-lfs dulu (untuk file model .pkl yang bisa > 10 MB)
git lfs install

# Clone repo Space (ganti USERNAME dan SPACE-NAME)
git clone https://huggingface.co/spaces/USERNAME/SPACE-NAME
cd SPACE-NAME
```

#### 3. Edit isi file

Setelah di-clone, struktur folder sudah ada. Yang perlu dilakukan:

**a. Ganti isi `src/streamlit_app.py`** dengan kode aplikasi kita (lihat section [Membuat Streamlit App](#membuat-streamlit-app)).

**b. Salin file model dari hasil notebook:**

```bash
cp /path/to/notebook/model_linear_regression.pkl .
cp /path/to/notebook/model_decision_tree.pkl .
```

**c. Update `requirements.txt`:**

```txt
streamlit==1.35.0
scikit-learn==1.5.0
joblib==1.4.2
numpy==1.26.4
pandas==2.2.2
```

**d. Update `Dockerfile`** — pastikan CMD menunjuk ke `src/streamlit_app.py` dan port 7860 (lihat section [Dockerfile](#dockerfile)).

**e. Update `README.md`** — pastikan header YAML-nya:

```markdown
---
title: House Price Predictor
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# House Price Predictor

Prediksi harga rumah di King County, Washington menggunakan
Linear Regression dan Decision Tree Regressor.
```

> ⚠️ `app_port` harus sama dengan `EXPOSE` dan `--server.port` di Dockerfile (7860).  
> Saat SDK Docker, field `sdk_version` dan `app_file` **tidak dipakai**.

#### 4. Login ke HuggingFace

Sebelum bisa push, harus autentikasi ke HuggingFace terlebih dahulu:

```bash
# Install HuggingFace CLI
pip install -U "huggingface_hub[cli]"

# Login — akan minta token (buat di huggingface.co > Settings > Access Tokens)
hf auth login
```

#### 5. Track model dengan git-lfs lalu push

File `.gitattributes` sudah ada dari template HuggingFace dan sudah men-track `*.pkl`. Tinggal commit dan push:

```bash
# Verifikasi .pkl sudah di-track git-lfs
cat .gitattributes

# Stage semua perubahan
git add .

# Commit
git commit -m "Add house price predictor app and models"

# Push — Space otomatis build ulang
git push
```

> ✅ **Selalu pin versi** di `requirements.txt` — tanpa pin versi, update library bisa merusak kompatibilitas model yang sudah disimpan.

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
