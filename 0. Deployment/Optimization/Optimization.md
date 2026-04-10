# Optimization

## Daftar Isi

1. [Apa itu Model Optimization?](#apa-itu-model-optimization)
2. [Mengapa Perlu Dioptimasi?](#mengapa-perlu-dioptimasi)
3. [Kuantisasi](#kuantisasi)
4. [Pruning](#pruning)
5. [ONNX Runtime](#onnx-runtime)
6. [Caching dan Batching](#caching-dan-batching)
7. [Perbandingan Teknik](#perbandingan-teknik)

---

## Apa itu Model Optimization?

**Model Optimization** adalah sekumpulan teknik untuk membuat model ML menjadi **lebih cepat**, **lebih kecil**, dan **lebih hemat memori** tanpa (atau dengan sedikit) mengorbankan akurasi — agar bisa berjalan efisien di produksi.

> **Analogi:** Model aslimu seperti mesin jet — powerful tapi boros bahan bakar dan butuh landasan panjang. Optimasi adalah mengubahnya jadi motor matic — cukup cepat untuk kebutuhan sehari-hari, jauh lebih hemat, dan bisa lewat gang sempit. 🛵 vs ✈️

---

## Mengapa Perlu Dioptimasi?

| Masalah Tanpa Optimasi | Dampaknya |
|------------------------|-----------|
| Model terlalu besar | Tidak muat di RAM server kecil / mobile |
| Inferensi lambat | Latensi tinggi, user experience buruk |
| Komputasi mahal | Biaya cloud meledak |
| Memori besar | Tidak bisa deploy di edge device / IoT |

---

## Kuantisasi

**Kuantisasi** mengurangi presisi angka dalam model — dari `float32` (32-bit) menjadi `int8` (8-bit) atau `float16`. Ukuran model bisa turun hingga **4x** dengan kehilangan akurasi minimal.

### Kuantisasi untuk scikit-learn (via ONNX)

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Konversi ke ONNX terlebih dahulu
initial_type = [('float_input', FloatTensorType([None, n_features]))]
onnx_model   = convert_sklearn(model, initial_types=initial_type)
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# 2. Kuantisasi model ONNX
quantize_dynamic(
    model_input='model.onnx',
    model_output='model_quantized.onnx',
    weight_type=QuantType.QInt8
)

# 3. Bandingkan ukuran
import os
original  = os.path.getsize('model.onnx') / 1024
quantized = os.path.getsize('model_quantized.onnx') / 1024
print(f"Original  : {original:.1f} KB")
print(f"Quantized : {quantized:.1f} KB")
print(f"Kompresi  : {original/quantized:.1f}x lebih kecil")
```

### Kuantisasi untuk PyTorch

```python
import torch

model.eval()

# Dynamic quantization (paling mudah)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Layer yang dikuantisasi
    dtype=torch.qint8
)

# Simpan
torch.save(quantized_model.state_dict(), 'model_quantized.pt')
print(f"Ukuran asli     : {os.path.getsize('model.pt') / 1e6:.2f} MB")
print(f"Ukuran quantized: {os.path.getsize('model_quantized.pt') / 1e6:.2f} MB")
```

---

## Pruning

**Pruning** menghapus bobot (weight) atau neuron yang kontribusinya kecil terhadap output — membuat model lebih sparse dan lebih kecil.

```python
import torch
import torch.nn.utils.prune as prune

# Prune 30% bobot terkecil pada layer Linear
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

# Cek sparsity
sparsity = float(torch.sum(model.fc1.weight == 0)) / model.fc1.weight.nelement()
print(f"Sparsity layer fc1: {sparsity:.1%}")

# Remove mask (jadikan permanen)
prune.remove(model.fc1, 'weight')

# Global pruning (prune 20% bobot terkecil dari semua layer sekaligus)
parameters_to_prune = [
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2
)
```

---

## ONNX Runtime

**ONNX Runtime** adalah inference engine yang bisa menjalankan model ONNX jauh lebih cepat dari implementasi Python biasa, karena dioptimasi di level C++.

```python
import onnxruntime as rt
import numpy as np
import time

# Load model ONNX
sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = rt.InferenceSession(
    'model.onnx',
    sess_options=sess_options,
    providers=['CPUExecutionProvider']  # atau 'CUDAExecutionProvider' untuk GPU
)

input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Benchmark: ONNX Runtime vs scikit-learn
X_bench = X_test.astype(np.float32)
N = 1000

# Scikit-learn
start = time.perf_counter()
for _ in range(N):
    model.predict(X_test)
time_sklearn = (time.perf_counter() - start) / N * 1000

# ONNX Runtime
start = time.perf_counter()
for _ in range(N):
    sess.run([output_name], {input_name: X_bench})
time_onnx = (time.perf_counter() - start) / N * 1000

print(f"Scikit-learn  : {time_sklearn:.3f} ms/request")
print(f"ONNX Runtime  : {time_onnx:.3f} ms/request")
print(f"Speedup       : {time_sklearn/time_onnx:.1f}x")
```

---

## Caching dan Batching

### Model Caching

Load model **sekali saja** saat aplikasi start, bukan setiap request.

```python
# Streamlit
import streamlit as st
import joblib

@st.cache_resource          # Cache di memori, tidak reload tiap request
def load_model():
    return joblib.load('pipeline.joblib')

pipeline = load_model()

# FastAPI — load di level modul (bukan di dalam fungsi endpoint)
from contextlib import asynccontextmanager
from fastapi import FastAPI

ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model['pipeline'] = joblib.load('pipeline.joblib')
    yield
    ml_model.clear()

app = FastAPI(lifespan=lifespan)
```

### Batch Inference

Proses banyak request sekaligus lebih efisien daripada satu-satu.

```python
import numpy as np

def predict_batch(list_of_features: list, batch_size: int = 32) -> list:
    """Prediksi dalam batch untuk efisiensi."""
    all_predictions = []

    for i in range(0, len(list_of_features), batch_size):
        batch   = np.array(list_of_features[i:i + batch_size])
        preds   = pipeline.predict(batch)
        all_predictions.extend(preds.tolist())

    return all_predictions
```

---

## Perbandingan Teknik

| Teknik | Pengurangan Ukuran | Speedup | Kehilangan Akurasi | Kompleksitas |
|--------|-------------------|---------|--------------------|-------------|
| **Kuantisasi (int8)** | 4x | 2-4x | Sangat kecil | Rendah |
| **Pruning (50%)** | 2x | 1.5-3x | Kecil (perlu fine-tune) | Sedang |
| **ONNX Runtime** | Tidak ada | 2-10x | Tidak ada | Rendah |
| **Knowledge Distillation** | 5-10x | 5-10x | Sedang | Tinggi |
| **Batch Inference** | Tidak ada | 5-20x (throughput) | Tidak ada | Rendah |

---

## Referensi

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Scikit-learn: Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
