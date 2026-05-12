# Forward & Backward Propagation

## Daftar Isi

- [Forward \& Backward Propagation](#forward--backward-propagation)
  - [Daftar Isi](#daftar-isi)
  - [Gambaran Umum](#gambaran-umum)
  - [Forward Propagation](#forward-propagation)
    - [Matematika Forward Pass](#matematika-forward-pass)
    - [Contoh Perhitungan Manual](#contoh-perhitungan-manual)
  - [Backward Propagation](#backward-propagation)
    - [Intuisi Backpropagation](#intuisi-backpropagation)
    - [Chain Rule](#chain-rule)
    - [Matematika Backpropagation](#matematika-backpropagation)
  - [Implementasi](#implementasi)
    - [Backprop Manual dari Scratch (NumPy)](#backprop-manual-dari-scratch-numpy)
    - [Forward \& Backward dengan PyTorch (Otomatis)](#forward--backward-dengan-pytorch-otomatis)
  - [Referensi](#referensi)

---

## Gambaran Umum

Training sebuah neural network pada dasarnya adalah proses **berulang** dua langkah:

1. **Forward Propagation** — data mengalir dari input ke output untuk menghasilkan prediksi
2. **Backward Propagation** — error mengalir balik dari output ke input untuk memperbarui bobot

Bayangkan kamu belajar memanah. Pertama, kamu melempar anak panah (forward pass) — hasilnya meleset. Lalu kamu menganalisis seberapa jauh melesetnya dan ke arah mana, kemudian menyesuaikan cara memegang busur (backward pass). Proses ini berulang sampai panahmu tepat sasaran.

---

## Forward Propagation

Forward propagation adalah proses menghitung prediksi model dari input menuju output, layer demi layer.

![forward pass](https://media.geeksforgeeks.org/wp-content/uploads/20260504170312160544/architecture-of-a-neural-network.webp)

### Matematika Forward Pass

Untuk setiap layer `l`, forward propagation melakukan dua operasi:

**Langkah 1 — Linear Transformation:**
```
Z[l] = W[l] · A[l-1] + b[l]
```

**Langkah 2 — Activation Function:**
```
A[l] = f(Z[l])
```

Di mana:
- `W[l]` = weight matrix layer ke-`l`
- `A[l-1]` = output (aktivasi) layer sebelumnya (untuk layer pertama, `A[0] = X` = input)
- `b[l]` = bias vector layer ke-`l`
- `f()` = activation function
- `Z[l]` = pre-activation value
- `A[l]` = post-activation value (output layer `l`)

### Contoh Perhitungan Manual

Misalkan kita punya jaringan sederhana dengan:
- 2 neuron input
- 1 hidden layer dengan 2 neuron (aktivasi ReLU)
- 1 output neuron (aktivasi Sigmoid)

```
Input: x = [1.0, 0.5]

Bobot layer 1:
W[1] = [[0.1, 0.4],
        [0.3, 0.2]]
b[1] = [0.1, 0.1]

Bobot layer 2:
W[2] = [[0.5, 0.7]]
b[2] = [0.2]
```

**Forward pass:**

```
Z[1] = W[1] · x + b[1]
     = [[0.1, 0.4], [0.3, 0.2]] · [1.0, 0.5] + [0.1, 0.1]
     = [0.1×1 + 0.4×0.5, 0.3×1 + 0.2×0.5] + [0.1, 0.1]
     = [0.3, 0.4] + [0.1, 0.1]
     = [0.4, 0.5]

A[1] = ReLU(Z[1]) = ReLU([0.4, 0.5]) = [0.4, 0.5]  ← positif semua, tidak berubah

Z[2] = W[2] · A[1] + b[2]
     = [0.5, 0.7] · [0.4, 0.5] + 0.2
     = 0.5×0.4 + 0.7×0.5 + 0.2
     = 0.2 + 0.35 + 0.2
     = 0.75

A[2] = Sigmoid(Z[2]) = 1 / (1 + e^(-0.75)) ≈ 0.679
```

Hasil prediksi: **0.679** (probabilitas kelas positif ≈ 67.9%)

---

## Backward Propagation

Backward propagation (atau backprop) adalah algoritma untuk menghitung **gradient** dari loss function terhadap setiap weight dan bias dalam jaringan. Gradient ini kemudian digunakan untuk memperbarui parameter model.

### Intuisi Backpropagation

Setelah mendapatkan prediksi, kita hitung **loss** (seberapa salah prediksi kita). Lalu kita tanya: "Kalau saya naikkan/turunkan weight ini sedikit, apakah loss akan berkurang?"

Jawaban atas pertanyaan itu adalah **gradient** — arah dan besaran perubahan yang diperlukan.

Backprop menghitung gradient ini secara efisien menggunakan **chain rule** dari kalkulus, dimulai dari layer output dan bergerak mundur ke input.

### Chain Rule

Chain rule menyatakan bahwa jika `y = f(g(x))`, maka:

```
dy/dx = (dy/dg) × (dg/dx)
```

Dalam konteks neural network: loss bergantung pada output, output bergantung pada Z, Z bergantung pada weight. Untuk menghitung `∂L/∂W`, kita kalikan gradient di sepanjang rantai tersebut.

### Matematika Backpropagation

Misalkan kita punya loss `L`. Backprop menghitung dari layer terakhir ke pertama:

**Output layer:**
```
∂L/∂Z[L] = A[L] - y   (untuk Binary Cross-Entropy loss)
```

**Untuk setiap layer l (dari akhir ke awal):**
```
∂L/∂W[l] = (1/m) × ∂L/∂Z[l] · A[l-1]ᵀ
∂L/∂b[l] = (1/m) × Σ ∂L/∂Z[l]
∂L/∂A[l-1] = W[l]ᵀ · ∂L/∂Z[l]
∂L/∂Z[l-1] = ∂L/∂A[l-1] × f'(Z[l-1])
```

Di mana:
- `m` = jumlah sampel dalam batch
- `f'` = turunan dari activation function
- `ᵀ` = transpose matriks

**Update parameter (Gradient Descent):**
```
W[l] = W[l] - α × ∂L/∂W[l]
b[l] = b[l] - α × ∂L/∂b[l]
```

Di mana `α` (alpha) adalah **learning rate** — seberapa besar langkah pembaruan yang diambil.

---

## Implementasi

### Backprop Manual dari Scratch (NumPy)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        # Inisialisasi weight dengan nilai kecil random
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
        self.lr = lr

    def forward(self, X):
        # Simpan nilai intermediate untuk backprop
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        m = X.shape[1]

        # Layer output (Sigmoid + BCE loss)
        dZ2 = self.A2 - y
        dW2 = (1/m) * dZ2 @ self.A1.T
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Layer hidden (ReLU)
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (1/m) * dZ1 @ X.T
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update parameter
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def compute_loss(self, y_pred, y):
        m = y.shape[1]
        # Binary Cross-Entropy Loss
        loss = -(1/m) * np.sum(y * np.log(y_pred + 1e-8) + (1-y) * np.log(1-y_pred + 1e-8))
        return loss

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### Forward & Backward dengan PyTorch (Otomatis)

PyTorch memiliki fitur **autograd** yang menghitung gradient secara otomatis:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Data dummy
X = torch.tensor([[1.0, 0.5], [0.3, 0.8], [0.9, 0.1]], dtype=torch.float32)
y = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

# Model sederhana
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

loss_fn  = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # Backward pass (PyTorch hitung gradient otomatis)
    optimizer.zero_grad()  # Reset gradient dari iterasi sebelumnya
    loss.backward()        # Hitung gradient
    optimizer.step()       # Update weights

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## Referensi

- [3Blue1Brown: Backpropagation, Step by Step](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [Deep Learning Book - Ian Goodfellow, Section 6.5: Back-Propagation](https://www.deeplearningbook.org/contents/mlp.html)
- [PyTorch Autograd: Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [CS231n: Backpropagation Intuitions](https://cs231n.github.io/optimization-2/)
