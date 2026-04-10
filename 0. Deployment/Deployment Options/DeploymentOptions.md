# Deployment Options

## Daftar Isi

1. [Apa itu Deployment Options?](#apa-itu-deployment-options)
2. [Kategori Deployment](#kategori-deployment)
3. [Flask](#flask)
4. [FastAPI](#fastapi)
5. [Streamlit](#streamlit)
6. [Gradio](#gradio)
7. [Perbandingan](#perbandingan)

---

## Apa itu Deployment Options?

Setelah model dilatih dan diserialisasi, ada banyak cara untuk membuatnya bisa diakses — mulai dari REST API untuk developer, hingga web app interaktif untuk end user. Pilihan tergantung pada **siapa yang akan menggunakan** dan **bagaimana cara penggunaannya**.

> **Analogi:** Model kamu adalah makanan yang sudah matang. Flask/FastAPI itu seperti layanan katering — kamu kirim pesanan (request), mereka kirim kembali makanan (response). Streamlit/Gradio itu seperti restoran — pengguna datang langsung dan memesan lewat meja (UI). 🍽️

---

## Kategori Deployment

| Kategori | Contoh | Untuk Siapa |
|----------|--------|-------------|
| **REST API** | Flask, FastAPI | Developer / sistem lain yang butuh endpoint |
| **Web App (No-code UI)** | Streamlit, Gradio | End user, demo, prototipe cepat |
| **Serverless** | AWS Lambda, Google Cloud Functions | Skala besar, bayar per request |
| **Container** | Docker + Kubernetes | Produksi enterprise, skalabilitas tinggi |
| **Edge / On-device** | ONNX Runtime, TFLite | Mobile, IoT, tanpa koneksi internet |
| **Platform ML** | HuggingFace Spaces, Replicate | Demo publik, sharing cepat |

---

## Flask

**Flask** adalah web framework Python yang ringan dan fleksibel, cocok untuk membuat REST API dari model ML.

```python
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load pipeline saat server start (bukan tiap request)
pipeline = joblib.load('pipeline.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'features' not in data:
        return jsonify({'error': 'Key "features" tidak ditemukan'}), 400

    features = np.array(data['features']).reshape(1, -1)
    prediction = pipeline.predict(features)[0]
    probability = pipeline.predict_proba(features)[0].tolist()

    return jsonify({
        'prediction': int(prediction),
        'probability': probability
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Contoh request:**

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## FastAPI

**FastAPI** adalah framework modern yang lebih cepat dari Flask dengan fitur **type validation otomatis** dan **dokumentasi Swagger** bawaan.

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="ML Model API", version="1.0")

pipeline = joblib.load('pipeline.joblib')

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: int
    probability: List[float]

@app.post('/predict', response_model=PredictResponse)
def predict(body: PredictRequest):
    features = np.array(body.features).reshape(1, -1)

    if features.shape[1] != pipeline.n_features_in_:
        raise HTTPException(
            status_code=422,
            detail=f"Diharapkan {pipeline.n_features_in_} fitur, dapat {features.shape[1]}"
        )

    prediction  = int(pipeline.predict(features)[0])
    probability = pipeline.predict_proba(features)[0].tolist()

    return PredictResponse(prediction=prediction, probability=probability)

@app.get('/health')
def health():
    return {'status': 'ok'}

# Jalankan: uvicorn main:app --host 0.0.0.0 --port 8000
# Dokumentasi: http://localhost:8000/docs
```

---

## Streamlit

**Streamlit** memungkinkan pembuatan web app interaktif hanya dari Python — tanpa HTML/CSS/JavaScript.

```python
# app.py
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="ML App", page_icon="🤖")
st.title("🤖 Prediksi Klasifikasi Iris")
st.write("Masukkan pengukuran bunga untuk memprediksi jenisnya.")

@st.cache_resource
def load_model():
    return joblib.load('pipeline.joblib')

pipeline     = load_model()
class_names  = ['Setosa', 'Versicolor', 'Virginica']

with st.sidebar:
    st.header("Input Fitur")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.5, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 0.2, 0.1)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Prediksi", type="primary"):
    prediction  = pipeline.predict(features)[0]
    probability = pipeline.predict_proba(features)[0]

    st.success(f"Prediksi: **{class_names[prediction]}**")

    st.subheader("Probabilitas")
    for name, prob in zip(class_names, probability):
        st.progress(float(prob), text=f"{name}: {prob:.1%}")
```

---

## Gradio

**Gradio** adalah alternatif Streamlit yang sangat mudah digunakan dan terintegrasi erat dengan HuggingFace.

```python
# app.py
import gradio as gr
import joblib
import numpy as np

pipeline    = joblib.load('pipeline.joblib')
class_names = ['Setosa', 'Versicolor', 'Virginica']

def predict(sepal_length, sepal_width, petal_length, petal_width):
    features    = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction  = pipeline.predict(features)[0]
    probability = pipeline.predict_proba(features)[0]
    return {class_names[i]: float(probability[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.1, label="Sepal Length"),
        gr.Slider(2.0, 4.5, value=3.5, label="Sepal Width"),
        gr.Slider(1.0, 7.0, value=1.4, label="Petal Length"),
        gr.Slider(0.1, 2.5, value=0.2, label="Petal Width"),
    ],
    outputs=gr.Label(num_top_classes=3),
    title="Iris Classifier",
    description="Prediksi jenis bunga iris dari ukurannya."
)

demo.launch()
```

---

## Perbandingan

| | Flask | FastAPI | Streamlit | Gradio |
|---|---|---|---|---|
| **Tipe** | REST API | REST API | Web App | Web App |
| **Target** | Developer | Developer | End User | End User |
| **Kemudahan** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Kecepatan** | Cukup | Sangat cepat | Sedang | Sedang |
| **Docs otomatis** | ❌ | ✅ Swagger | N/A | ✅ |
| **HF Spaces** | ⚠️ Manual | ⚠️ Manual | ✅ Native | ✅ Native |
| **Cocok untuk** | Backend API | Backend API | Demo, prototipe | Demo cepat, ML |

---

## Referensi

- [Flask Documentation](https://flask.palletsprojects.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)
