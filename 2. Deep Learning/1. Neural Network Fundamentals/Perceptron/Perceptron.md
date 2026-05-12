# Perceptron & Multi-Layer Perceptron

## Daftar Isi

- [Perceptron \& Multi-Layer Perceptron](#perceptron--multi-layer-perceptron)
  - [Daftar Isi](#daftar-isi)
  - [Apa itu Perceptron?](#apa-itu-perceptron)
    - [Analogi Sederhananya](#analogi-sederhananya)
  - [Model Matematis Perceptron](#model-matematis-perceptron)
    - [Peran Bias](#peran-bias)
    - [Visualisasi Perceptron](#visualisasi-perceptron)
  - [Keterbatasan Perceptron](#keterbatasan-perceptron)
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
    - [Struktur MLP](#struktur-mlp)
    - [Bagaimana MLP Menyelesaikan XOR?](#bagaimana-mlp-menyelesaikan-xor)
    - [Parameter MLP](#parameter-mlp)
  - [Implementasi](#implementasi)
    - [Perceptron dari Scratch (NumPy)](#perceptron-dari-scratch-numpy)
    - [MLP dengan PyTorch](#mlp-dengan-pytorch)
  - [Kelebihan \& Kekurangan](#kelebihan--kekurangan)
  - [Referensi](#referensi)

---

## Apa itu Perceptron?

Perceptron adalah unit komputasi paling dasar dalam neural network — satu neuron buatan yang menerima beberapa input, memprosesnya, dan menghasilkan satu output.

Konsep ini pertama kali diperkenalkan oleh **Frank Rosenblatt** pada tahun 1958, terinspirasi dari cara neuron biologis bekerja di otak manusia.

### Analogi Sederhananya

Bayangkan kamu harus memutuskan apakah akan pergi ke pantai hari ini atau tidak. Kamu mempertimbangkan beberapa faktor:

- Apakah cuacanya cerah? (bobot tinggi, sangat berpengaruh)
- Apakah kamu punya waktu luang? (bobot sedang)
- Apakah ada teman yang mau ikut? (bobot rendah)

Kamu menjumlahkan semua pertimbangan itu, dan kalau totalnya melewati ambang batas tertentu — kamu berangkat. Kalau tidak — kamu di rumah.

Perceptron bekerja persis seperti itu. Input dikalikan bobot (weights), dijumlahkan, lalu dibandingkan dengan threshold untuk menghasilkan keputusan akhir.

---

## Model Matematis Perceptron

Secara matematis, sebuah perceptron melakukan operasi berikut:

**Langkah 1 — Weighted Sum:**

```
z = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + b
```

Di mana:
- `x₁, x₂, ..., xₙ` adalah input
- `w₁, w₂, ..., wₙ` adalah bobot (weights) — seberapa penting tiap input
- `b` adalah bias — nilai tambahan agar model lebih fleksibel
- `z` adalah hasil penjumlahan berbobot

**Langkah 2 — Activation Function:**

```
output = f(z)
```

Pada perceptron klasik, fungsi aktivasi yang digunakan adalah **step function**:

```
f(z) = 1 jika z >= 0
f(z) = 0 jika z < 0
```

### Peran Bias

Bias (`b`) adalah nilai tambahan yang tidak bergantung pada input. Fungsinya mirip seperti konstanta dalam persamaan garis — ia menggeser batas keputusan model sehingga tidak harus melewati titik origin. Tanpa bias, kemampuan model sangat terbatas.

### Visualisasi Perceptron

```
x₁ ──[w₁]──┐
            │
x₂ ──[w₂]──┤──► Σ (z = Σwᵢxᵢ + b) ──► f(z) ──► output
            │
x₃ ──[w₃]──┘
       ↑
      [b] (bias)
```

---

## Keterbatasan Perceptron

Perceptron hanya bisa memisahkan data yang **linearly separable** — artinya, data yang bisa dipisahkan oleh satu garis lurus.

Masalah paling terkenal yang tidak bisa diselesaikan perceptron tunggal adalah **XOR**:

| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

Tidak ada satu garis lurus yang bisa memisahkan output 0 dan 1 pada tabel di atas. Ini yang mendorong lahirnya **Multi-Layer Perceptron**.

---

## Multi-Layer Perceptron (MLP)

Multi-Layer Perceptron (MLP) adalah perluasan dari perceptron dengan menambahkan satu atau lebih **hidden layer** di antara input dan output.

### Struktur MLP

```
Input Layer    Hidden Layer    Output Layer
    x₁  ─────── h₁ ──────────── ŷ₁
    x₂  ─────── h₂ ──────────── ŷ₂
    x₃  ─────── h₃
                h₄
```

MLP terdiri dari tiga jenis layer:

| Layer | Fungsi |
|---|---|
| **Input Layer** | Menerima data mentah (fitur/feature) |
| **Hidden Layer** | Memproses dan mengekstraksi representasi dari data |
| **Output Layer** | Menghasilkan prediksi akhir |

### Bagaimana MLP Menyelesaikan XOR?

Dengan menambahkan hidden layer, MLP bisa membuat **multiple decision boundaries** — bukan hanya satu garis, tapi kombinasi batas yang lebih kompleks. Hidden layer pertama membuat dua garis pemisah, lalu output layer menggabungkan hasilnya untuk mengklasifikasikan XOR dengan benar.

### Parameter MLP

MLP memiliki dua jenis parameter yang dipelajari selama training:
- **Weights (W)**: bobot koneksi antar neuron
- **Bias (b)**: nilai tambahan di setiap neuron

Jumlah total parameter bergantung pada jumlah layer dan neuron di setiap layer. Semakin banyak parameter, semakin besar kapasitas model untuk mempelajari pola kompleks — tapi juga semakin rentan terhadap overfitting.

---

## Implementasi

### Perceptron dari Scratch (NumPy)

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                z = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if z >= 0 else 0
                
                # Update weights hanya jika prediksi salah
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.where(z >= 0, 1, 0)

# Contoh penggunaan: klasifikasi AND
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])  # AND gate

model = Perceptron(learning_rate=0.1, n_epochs=10)
model.fit(X, y)
print(model.predict(X))  # Output: [0, 0, 0, 1]
```

### MLP dengan PyTorch

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Inisialisasi model
model = MLP(input_size=2, hidden_size=4, output_size=1)
print(model)

# Hitung jumlah parameter
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameter: {total_params}")
```

---

## Kelebihan & Kekurangan

| | Kelebihan | Kekurangan |
|---|---|---|
| **Perceptron** | Sederhana, cepat, mudah dipahami | Hanya bisa klasifikasi linear, tidak bisa XOR |
| **MLP** | Bisa mempelajari pola non-linear, fleksibel | Butuh banyak data, komputasi lebih berat, rentan overfitting |

---

## Referensi

- [Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://psycnet.apa.org/record/1959-09865-001)
- [Deep Learning Book - Ian Goodfellow, Chapter 6](https://www.deeplearningbook.org/contents/mlp.html)
- [3Blue1Brown: But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [PyTorch: nn.Linear Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
