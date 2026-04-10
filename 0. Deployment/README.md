# Deployment

Deployment adalah proses membawa model machine learning dari lingkungan eksperimen (notebook/lab) ke lingkungan produksi yang bisa diakses oleh pengguna nyata. Tanpa deployment, model hanya berguna di komputer kita sendiri.

## Struktur Modul

| Modul | Deskripsi |
|-------|-----------|
| 📦 [Serialization](./Serialization/Serialization.md) | Menyimpan dan memuat model (pickle, joblib, ONNX) |
| 🚀 [Deployment Options](./Deployment%20Options/DeploymentOptions.md) | Flask, FastAPI, Streamlit, dan opsi lainnya |
| ⚡ [Optimization](./Optimization/Optimization.md) | Kuantisasi, pruning, ONNX Runtime, dan teknik optimasi lainnya |
| 🤗 [HuggingFace Spaces](./HuggingFace%20Spaces/HuggingFaceSpaces.md) | Deploy model ke HF Spaces dengan Streamlit |

## Alur Deployment

```
Data → Preprocessing → Training → Evaluasi → Serialization → Serving → Monitoring
         ↑                                         ↓
         └─────────── Retraining ←─────────────────┘
```

## Checklist Sebelum Deploy

1. Model sudah dievaluasi dan performanya memadai
2. Model sudah diserialisasi dengan benar (termasuk preprocessor)
3. Dependencies sudah di-pin ke versi tertentu (`requirements.txt`)
4. Endpoint sudah diuji dengan input edge case
5. Monitoring dan logging sudah disiapkan
