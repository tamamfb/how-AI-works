# Artificial Neural Network (ANN)

## Daftar Isi

1. [Apa itu ANN?](#apa-itu-ann)
2. [Struktur Jaringan](#struktur-jaringan)
3. [Fungsi Aktivasi](#fungsi-aktivasi)
4. [Proses Training](#proses-training)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu ANN?

**Artificial Neural Network (ANN)** adalah model komputasi yang terinspirasi dari cara kerja **otak manusia**. Terdiri dari neuron buatan yang saling terhubung dalam lapisan-lapisan, setiap neuron memproses informasi dan meneruskannya ke neuron berikutnya.

> **Analogi:** Bayangin tim relay lari — setiap pelari (neuron) menerima tongkat (sinyal), mengolahnya sedikit, lalu meneruskannya ke pelari berikutnya. Setelah melalui semua lapisan pelari, hasil akhirnya adalah prediksi. Semakin banyak lapisan, semakin kompleks pola yang bisa dipelajari! 🏃‍♂️🏃‍♀️

---

## Struktur Jaringan

ANN tersusun dari tiga jenis lapisan:

| Lapisan | Fungsi |
|---------|--------|
| **Input Layer** | Menerima data mentah — jumlah neuron = jumlah fitur |
| **Hidden Layer(s)** | Memproses dan mengekstrak pola dari data |
| **Output Layer** | Menghasilkan prediksi — jumlah neuron = jumlah kelas |

Setiap neuron melakukan operasi:

$$z = \sum_{i} w_i x_i + b \quad \xrightarrow{\text{aktivasi}} \quad a = f(z)$$

Di mana $w_i$ adalah bobot, $b$ adalah bias, dan $f$ adalah fungsi aktivasi.

---

## Fungsi Aktivasi

Fungsi aktivasi memperkenalkan **non-linearitas** ke dalam jaringan, memungkinkan ANN mempelajari pola kompleks:

| Fungsi | Rumus | Dipakai di |
|--------|-------|-----------|
| **ReLU** | $\max(0, z)$ | Hidden layer (paling umum) |
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | Output biner |
| **Softmax** | $\frac{e^{z_k}}{\sum e^{z_j}}$ | Output multi-kelas |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | Hidden layer (nilai -1 sampai 1) |
| **Leaky ReLU** | $\max(0.01z, z)$ | Alternatif ReLU, cegah dying neurons |

---

## Proses Training

ANN dilatih melalui proses **Backpropagation** + **Gradient Descent**:

1. **Forward Pass** — input diproses maju melalui semua lapisan hingga menghasilkan output
2. **Hitung Loss** — ukur kesalahan prediksi dengan fungsi loss (Cross-Entropy untuk klasifikasi)
3. **Backward Pass** — hitung gradien loss terhadap setiap bobot menggunakan chain rule
4. **Update Bobot** — sesuaikan bobot menggunakan optimizer (SGD, Adam, RMSprop)
5. **Ulangi** hingga loss konvergen

**Loss untuk klasifikasi multi-kelas:**

$$L = -\sum_{k=1}^K y_k \log(\hat{p}_k)$$

---

## Implementasi

### Menggunakan Scikit-learn (MLPClassifier)

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Scaling WAJIB untuk ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 2. Inisialisasi dan training
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # 2 hidden layer: 128 dan 64 neuron
    activation='relu',             # Fungsi aktivasi hidden layer
    solver='adam',                 # Optimizer
    alpha=0.0001,                  # Regularisasi L2
    learning_rate_init=0.001,
    max_iter=300,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# 3. Prediksi dan evaluasi
y_pred = mlp.predict(X_test_scaled)
print(f"Akurasi : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Menggunakan PyTorch

```python
import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model     = ANN(input_dim=X_train.shape[1], hidden_dim=128, output_dim=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss    = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Bisa belajar pola sangat kompleks dan non-linear | Membutuhkan banyak data dan komputasi |
| Fleksibel untuk berbagai jenis data | Sulit diinterpretasi — black box |
| State-of-the-art untuk banyak tugas | Banyak hyperparameter yang perlu di-tuning |
| Bisa diperluas menjadi CNN, RNN, Transformer | Rentan overfit tanpa regularisasi (Dropout, L2) |

---

## Referensi

- [Scikit-learn: MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [StatQuest: Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw)
