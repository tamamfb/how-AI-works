# Serialization

## Daftar Isi

1. [Apa itu Serialization?](#apa-itu-serialization)
2. [Pickle](#pickle)
3. [Joblib](#joblib)
4. [ONNX](#onnx)
5. [Menyimpan Pipeline Lengkap](#menyimpan-pipeline-lengkap)
6. [Perbandingan Format](#perbandingan-format)

---

## Apa itu Serialization?

**Serialization** adalah proses mengubah objek model (yang ada di memori RAM) menjadi **format yang bisa disimpan ke disk** dan dimuat kembali nanti — tanpa harus melatih ulang dari awal.

> **Analogi:** Bayangin kamu sudah susah payah merakit puzzle 1000 keping. Daripada berantakin dan rakit ulang besok, kamu foto hasilnya dan simpan. Besok tinggal lihat foto itu. Itulah serialization — "foto" dari model yang sudah dilatih! 📷

---

## Pickle

**Pickle** adalah library bawaan Python untuk serialisasi objek Python apapun, termasuk model scikit-learn.

```python
import pickle

# ---- Menyimpan model ----
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# ---- Memuat model ----
with open('model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)

# Prediksi langsung
y_pred = model_loaded.predict(X_test)
```

> ⚠️ **Peringatan:** File `.pkl` bisa mengeksekusi kode berbahaya saat dimuat. Jangan load file pickle dari sumber yang tidak dipercaya!

---

## Joblib

**Joblib** adalah alternatif pickle yang direkomendasikan untuk scikit-learn, terutama untuk model dengan **array numpy besar** (lebih efisien dalam kompresi dan kecepatan).

```python
import joblib

# ---- Menyimpan model ----
joblib.dump(model, 'model.joblib')

# Dengan kompresi (lebih kecil, sedikit lebih lambat)
joblib.dump(model, 'model.joblib', compress=3)  # compress: 0-9

# ---- Memuat model ----
model_loaded = joblib.load('model.joblib')

# Prediksi
y_pred = model_loaded.predict(X_test)
```

### Menyimpan Banyak Objek

```python
# Simpan model, scaler, dan label encoder sekaligus
artifacts = {
    'model': model,
    'scaler': scaler,
    'label_encoder': le,
    'feature_names': feature_names
}
joblib.dump(artifacts, 'artifacts.joblib')

# Memuat
artifacts = joblib.load('artifacts.joblib')
model   = artifacts['model']
scaler  = artifacts['scaler']
```

---

## ONNX

**ONNX (Open Neural Network Exchange)** adalah format terbuka yang memungkinkan model dari satu framework (scikit-learn, PyTorch, TensorFlow) dijalankan di framework lain atau runtime yang lebih cepat.

```python
# pip install skl2onnx onnxruntime

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np

# ---- Konversi model scikit-learn ke ONNX ----
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model   = convert_sklearn(model, initial_types=initial_type)

with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# ---- Inferensi dengan ONNX Runtime ----
sess  = rt.InferenceSession('model.onnx')
input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

X_test_f32 = X_test.astype(np.float32)
y_pred = sess.run([output_name], {input_name: X_test_f32})[0]
```

---

## Menyimpan Pipeline Lengkap

Yang paling penting adalah menyimpan **seluruh pipeline** — bukan hanya model, tapi juga preprocessor. Jika hanya simpan model, maka saat inferensi data baru harus dipreprocess ulang secara manual (rawan error).

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---- Buat dan latih pipeline ----
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

# ---- Simpan SELURUH pipeline ----
joblib.dump(pipeline, 'pipeline.joblib')

# ---- Memuat dan prediksi ----
# Data raw (belum di-scale) bisa langsung diproses!
pipeline_loaded = joblib.load('pipeline.joblib')
y_pred = pipeline_loaded.predict(X_test_raw)
```

> ✅ **Best practice:** Selalu simpan pipeline lengkap (preprocessor + model) dalam satu file. Ini memastikan inferensi konsisten dengan training.

---

## Perbandingan Format

| Format | Ukuran | Kecepatan Load | Lintas Framework | Keamanan | Cocok untuk |
|--------|--------|---------------|-----------------|---------|------------|
| **Pickle** | Sedang | Cepat | ❌ Python only | ⚠️ Rendah | Prototyping cepat |
| **Joblib** | Kecil (dengan kompresi) | Cepat | ❌ Python only | ⚠️ Rendah | Scikit-learn, array besar |
| **ONNX** | Kecil | Sangat cepat | ✅ Multi-platform | ✅ Aman | Produksi, inferensi cepat |
| **SavedModel (TF)** | Besar | Sedang | ✅ TF Serving | ✅ Aman | Model TensorFlow/Keras |
| **TorchScript** | Sedang | Cepat | ✅ C++, mobile | ✅ Aman | Model PyTorch produksi |

---

## Referensi

- [Scikit-learn: Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [ONNX: Open Neural Network Exchange](https://onnx.ai/)
- [Joblib Documentation](https://joblib.readthedocs.io/)
