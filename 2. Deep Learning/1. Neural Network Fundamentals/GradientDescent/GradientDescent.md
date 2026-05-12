# Gradient Descent & Variants

## Daftar Isi

- [Apa itu Gradient Descent?](#apa-itu-gradient-descent)
- [Variasi Gradient Descent](#variasi-gradient-descent)
  - [Batch Gradient Descent](#batch-gradient-descent)
  - [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
  - [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
- [Optimizer Modern](#optimizer-modern)
  - [Momentum](#momentum)
  - [RMSProp](#rmsprop)
  - [Adam](#adam)
- [Learning Rate dan Dampaknya](#learning-rate-dan-dampaknya)
- [Perbandingan Optimizer](#perbandingan-optimizer)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

---

## Apa itu Gradient Descent?

Gradient Descent adalah algoritma optimasi yang digunakan untuk **meminimalkan loss function** dengan secara iteratif memperbarui parameter model (weight dan bias) ke arah yang mengurangi nilai loss.

### Analogi Sederhananya

Bayangkan kamu berdiri di sebuah pegunungan berkabut dan ingin mencapai lembah terendah. Kamu tidak bisa melihat keseluruhan peta, tapi kamu bisa merasakan kemiringan tanah di sekitarmu. Caramu turun: lihat ke mana arah yang paling curam menurun di titikmu berdiri, ambil selangkah ke sana, evaluasi ulang, ulangi.

Gradient descent bekerja persis seperti itu. "Peta" adalah loss landscape, "posisimu" adalah nilai parameter, "kemiringan tanah" adalah gradient, dan "selangkah" adalah learning rate.

### Formula Dasar

```
θ = θ - α × ∂L/∂θ
```

Di mana:
- `θ` = parameter model (weight atau bias)
- `α` = learning rate (ukuran langkah)
- `∂L/∂θ` = gradient loss terhadap parameter

Kita **mengurangi** nilai θ dengan gradient-nya karena gradient menunjukkan arah naiknya loss — kita ingin bergerak ke arah sebaliknya.

---

## Variasi Gradient Descent

Perbedaan utama antara variasi GD terletak pada **berapa banyak data** yang digunakan untuk menghitung gradient di setiap langkah update.

### Batch Gradient Descent

Menggunakan **seluruh dataset** untuk menghitung gradient sebelum satu kali update:

```python
for epoch in range(n_epochs):
    gradient = compute_gradient(X_full, y_full, theta)
    theta = theta - lr * gradient
```

**Keunggulan:** Update stabil, konvergen ke minimum dengan baik.
**Kelemahan:** Sangat lambat dan membutuhkan memori besar untuk dataset besar.

---

### Stochastic Gradient Descent (SGD)

Menggunakan **satu sampel saja** untuk menghitung gradient di setiap update:

```python
for epoch in range(n_epochs):
    for x_i, y_i in zip(X, y):  # Satu sampel sekaligus
        gradient = compute_gradient(x_i, y_i, theta)
        theta = theta - lr * gradient
```

**Keunggulan:** Sangat cepat per iterasi, bisa "keluar" dari local minimum karena update-nya noisy.
**Kelemahan:** Update sangat tidak stabil (fluktuatif), bisa tidak konvergen dengan baik.

---

### Mini-Batch Gradient Descent

Kompromi terbaik — menggunakan **sebagian kecil data** (batch) per update:

```python
batch_size = 32
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        gradient = compute_gradient(X_batch, y_batch, theta)
        theta = theta - lr * gradient
```

**Keunggulan:** Keseimbangan antara stabilitas dan kecepatan; efisien di GPU karena bisa diparalelkan.
**Kelemahan:** Menambah hyperparameter baru: batch_size.

Ini adalah metode yang paling umum digunakan dalam deep learning modern. Ketika orang menyebut "SGD" dalam deep learning, biasanya yang dimaksud adalah mini-batch SGD.

---

## Optimizer Modern

SGD standar punya masalah: ia memperlakukan semua parameter dengan learning rate yang sama, dan tidak "mengingat" arah update sebelumnya. Optimizer modern mengatasi ini.

### Momentum

Momentum menambahkan "kecepatan" pada update — ia mengingat arah update sebelumnya dan terus bergerak ke arah itu, sehingga konvergensi lebih cepat dan stabil.

**Formula:**
```
v = β × v + (1-β) × ∂L/∂θ
θ = θ - α × v
```

Di mana:
- `v` = velocity (momentum)
- `β` = momentum coefficient (biasanya 0.9)

Analoginya seperti bola bowling yang bergulir turun bukit — ia makin cepat karena momentum, bukan berhenti di setiap cekungan kecil.

---

### RMSProp

RMSProp mengadaptasi learning rate secara individual untuk setiap parameter. Parameter yang gradientnya sering besar mendapat learning rate lebih kecil; yang jarang mendapat update mendapat learning rate lebih besar.

**Formula:**
```
s = β × s + (1-β) × (∂L/∂θ)²
θ = θ - (α / √(s + ε)) × ∂L/∂θ
```

Di mana:
- `s` = moving average dari kuadrat gradient
- `ε` = nilai kecil untuk stabilitas numerik (biasanya 1e-8)

RMSProp sangat efektif untuk data yang sparse (banyak fitur bernilai nol) dan untuk RNN.

---

### Adam

Adam (Adaptive Moment Estimation) adalah kombinasi dari Momentum dan RMSProp. Ia adalah optimizer paling populer dalam deep learning saat ini.

**Formula:**
```
m = β₁ × m + (1-β₁) × ∂L/∂θ         ← first moment (seperti Momentum)
v = β₂ × v + (1-β₂) × (∂L/∂θ)²      ← second moment (seperti RMSProp)

m̂ = m / (1 - β₁ᵗ)                    ← bias correction
v̂ = v / (1 - β₂ᵗ)                    ← bias correction

θ = θ - α × m̂ / (√v̂ + ε)
```

**Nilai default yang umum digunakan:**
- `β₁ = 0.9`
- `β₂ = 0.999`
- `ε = 1e-8`
- `α = 0.001`

Bias correction (`m̂` dan `v̂`) penting di awal training karena `m` dan `v` diinisialisasi dengan 0, yang menyebabkan estimasi yang bias.

---

## Learning Rate dan Dampaknya

Learning rate (`α`) adalah hyperparameter yang paling berpengaruh dalam training.

```
Learning Rate Terlalu Besar:
  Loss: ↑↑↓↑↑↓↑↑ (tidak stabil, mungkin divergen)

Learning Rate Ideal:
  Loss: ↓↓↓↓↓↓↓↓ (turun konsisten)

Learning Rate Terlalu Kecil:
  Loss: ↓ ↓ ↓ ↓ ↓ ↓ (turun sangat lambat)
```

Tidak ada nilai learning rate yang universal — ia harus disesuaikan dengan arsitektur model, dataset, dan optimizer yang digunakan.

---

## Perbandingan Optimizer

| Optimizer | Kecepatan | Stabilitas | Adaptif | Kapan Digunakan |
|---|---|---|---|---|
| **Batch GD** | Lambat | Sangat stabil | Tidak | Dataset kecil |
| **SGD** | Cepat | Tidak stabil | Tidak | Jarang digunakan sendiri |
| **SGD + Momentum** | Cepat | Cukup stabil | Tidak | Computer Vision, fine-tuning |
| **RMSProp** | Cepat | Stabil | Ya | RNN, data sparse |
| **Adam** | Cepat | Stabil | Ya | Default untuk kebanyakan kasus |
| **AdamW** | Cepat | Stabil | Ya | Transformer, NLP modern |

---

## Implementasi

### Implementasi Adam dari Scratch

```python
import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Timestep

    def update(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        self.t += 1
        updated_params = {}

        for key in params:
            # Update moment estimates
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Parameter update
            updated_params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return updated_params
```

### Semua Optimizer dengan PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

# Berbagai optimizer
sgd_optim      = optim.SGD(model.parameters(), lr=0.01)
momentum_optim = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
rmsprop_optim  = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
adam_optim     = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
adamw_optim    = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Contoh training loop dengan Adam
optimizer = adam_optim
criterion = nn.MSELoss()

X = torch.randn(100, 10)
y = torch.randn(100, 1)

for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")
```

---

## Referensi

- [An Overview of Gradient Descent Optimization Algorithms — Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/)
- [Adam: A Method for Stochastic Optimization — Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- [Deep Learning Book - Ian Goodfellow, Chapter 8: Optimization](https://www.deeplearningbook.org/contents/optimization.html)
- [PyTorch: torch.optim](https://pytorch.org/docs/stable/optim.html)
- [CS231n: Neural Networks Part 3 — Learning and Evaluation](https://cs231n.github.io/neural-networks-3/)
