# Loss Functions

## Daftar Isi

- [Apa itu Loss Function?](#apa-itu-loss-function)
- [Loss Function untuk Regresi](#loss-function-untuk-regresi)
  - [Mean Squared Error (MSE)](#mean-squared-error-mse)
  - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
  - [Huber Loss](#huber-loss)
- [Loss Function untuk Klasifikasi](#loss-function-untuk-klasifikasi)
  - [Binary Cross-Entropy](#binary-cross-entropy)
  - [Categorical Cross-Entropy](#categorical-cross-entropy)
- [Memilih Loss Function yang Tepat](#memilih-loss-function-yang-tepat)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

---

## Apa itu Loss Function?

Loss function (atau cost function) adalah cara model mengukur **seberapa jauh prediksinya dari jawaban yang benar**. Nilai loss yang besar berarti prediksi buruk; nilai kecil berarti prediksi baik.

Selama training, tujuan model adalah **meminimalkan nilai loss** — inilah yang menjadi landasan seluruh proses pembelajaran.

### Analogi Sederhananya

Bayangkan kamu belajar melempar darts. Setiap lemparan, kamu mengukur jarak antara posisi dart dengan pusat target. Jarak itulah loss-nya. Kamu terus berlatih sampai jarak rata-ratanya sekecil mungkin. Loss function dalam neural network bekerja persis seperti itu.

### Notasi

Misalkan:
- `y` = nilai aktual (ground truth label)
- `ŷ` = prediksi model
- `m` = jumlah sampel

---

## Loss Function untuk Regresi

Digunakan saat output model berupa nilai kontinu (bilangan real), misalnya memprediksi harga rumah, suhu, atau berat badan.

### Mean Squared Error (MSE)

**Formula:**
```
MSE = (1/m) × Σ (ŷᵢ - yᵢ)²
```

MSE menghitung rata-rata dari kuadrat perbedaan antara prediksi dan nilai aktual. Karena error dikuadratkan, MSE sangat sensitif terhadap **outlier** — satu prediksi yang meleset jauh bisa mendominasi nilai loss.

**Kapan digunakan:** Saat distribusi error diasumsikan normal (Gaussian) dan outlier ingin dihukum lebih berat.

**Kelemahan:** Tidak robust terhadap outlier.

---

### Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/m) × Σ |ŷᵢ - yᵢ|
```

MAE menggunakan nilai absolut dari perbedaan, bukan kuadrat. Hasilnya lebih **robust terhadap outlier** karena satu outlier tidak bisa mendominasi nilai loss. Namun, MAE tidak dapat didiferensiasi di titik nol (saat prediksi sama persis dengan target), yang sedikit mempersulit optimasi.

**Kapan digunakan:** Saat data mengandung banyak outlier.

---

### Huber Loss

**Formula:**
```
L_δ(y, ŷ) = (1/2)(ŷ - y)²                    jika |ŷ - y| ≤ δ
L_δ(y, ŷ) = δ × |ŷ - y| - (1/2)δ²           jika |ŷ - y| > δ
```

Huber Loss adalah kombinasi terbaik dari MSE dan MAE:
- Untuk error kecil (di bawah threshold δ): berperilaku seperti MSE (smooth, mudah dioptimasi)
- Untuk error besar (di atas threshold δ): berperilaku seperti MAE (robust terhadap outlier)

**Kapan digunakan:** Default yang baik untuk kebanyakan masalah regresi.

---

## Loss Function untuk Klasifikasi

Digunakan saat output model adalah kategori atau kelas, misalnya klasifikasi email (spam/bukan) atau pengenalan digit.

### Binary Cross-Entropy

**Formula:**
```
BCE = -(1/m) × Σ [yᵢ log(ŷᵢ) + (1 - yᵢ) log(1 - ŷᵢ)]
```

Digunakan untuk **klasifikasi biner** (dua kelas: 0 atau 1). Model menggunakan activation function Sigmoid di output layer untuk menghasilkan nilai probabilitas antara 0 dan 1.

**Bagaimana cara kerjanya:**
- Jika label aktual `y = 1` dan model memprediksi `ŷ = 0.9` (yakin benar): loss kecil ≈ 0.105
- Jika label aktual `y = 1` dan model memprediksi `ŷ = 0.1` (sangat salah): loss besar ≈ 2.303

Cross-entropy menghukum prediksi yang percaya diri tapi salah dengan sangat berat, mendorong model untuk lebih hati-hati.

---

### Categorical Cross-Entropy

**Formula:**
```
CCE = -(1/m) × Σ Σ yᵢₖ × log(ŷᵢₖ)
```

Di mana `K` adalah jumlah kelas dan `yᵢₖ` adalah 1 jika sampel `i` termasuk kelas `k`, 0 lainnya.

Digunakan untuk **klasifikasi multi-kelas** (lebih dari 2 kelas). Model menggunakan activation function Softmax di output layer.

Ini adalah generalisasi dari Binary Cross-Entropy untuk lebih dari dua kelas.

**Contoh:**
```
Label aktual (one-hot): y = [0, 1, 0]  → kelas 2 (anjing)
Prediksi Softmax:       ŷ = [0.1, 0.7, 0.2]

Loss = -(0×log(0.1) + 1×log(0.7) + 0×log(0.2))
     = -log(0.7)
     ≈ 0.357
```

Semakin tinggi probabilitas yang diberikan model untuk kelas yang benar, semakin kecil loss-nya.

---

## Memilih Loss Function yang Tepat

| Tipe Problem | Loss Function | Activation Output |
|---|---|---|
| Regresi (umum) | MSE | Linear (tidak ada) |
| Regresi (dengan outlier) | MAE atau Huber | Linear |
| Klasifikasi Biner | Binary Cross-Entropy | Sigmoid |
| Klasifikasi Multi-Kelas | Categorical Cross-Entropy | Softmax |

---

## Implementasi

### Perbandingan Loss Functions dengan PyTorch

```python
import torch
import torch.nn as nn

# Data dummy
y_true_reg   = torch.tensor([3.0, 5.0, 2.0, 8.0])
y_pred_reg   = torch.tensor([2.5, 5.5, 1.8, 7.0])

y_true_bin   = torch.tensor([1.0, 0.0, 1.0, 1.0])
y_pred_bin   = torch.tensor([0.9, 0.1, 0.8, 0.6])

y_true_cat   = torch.tensor([1, 2, 0])  # Label kelas
y_pred_cat   = torch.tensor([[0.1, 0.7, 0.2],
                               [0.1, 0.2, 0.7],
                               [0.8, 0.1, 0.1]])

# --- Regresi ---
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
huber_loss = nn.HuberLoss(delta=1.0)

print("=== Regresi ===")
print(f"MSE Loss:   {mse_loss(y_pred_reg, y_true_reg):.4f}")
print(f"MAE Loss:   {mae_loss(y_pred_reg, y_true_reg):.4f}")
print(f"Huber Loss: {huber_loss(y_pred_reg, y_true_reg):.4f}")

# --- Klasifikasi Biner ---
bce_loss = nn.BCELoss()
print("\n=== Klasifikasi Biner ===")
print(f"BCE Loss: {bce_loss(y_pred_bin, y_true_bin):.4f}")

# --- Klasifikasi Multi-Kelas ---
ce_loss = nn.CrossEntropyLoss()  # Di PyTorch, ini sudah termasuk Softmax
print("\n=== Klasifikasi Multi-Kelas ===")
print(f"Cross-Entropy Loss: {ce_loss(y_pred_cat, y_true_cat):.4f}")
```

### Loss Function dalam Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model untuk klasifikasi multi-kelas (misalnya 3 kelas)
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)   # Tidak perlu Softmax; CrossEntropyLoss sudah termasuk
)

# CrossEntropyLoss = Softmax + Negative Log Likelihood
criterion = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=0.001)

# Dummy data
X = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))

# Training loop
for epoch in range(200):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")
```

---

## Referensi

- [Deep Learning Book - Ian Goodfellow, Section 6.2: Output Units](https://www.deeplearningbook.org/contents/mlp.html)
- [PyTorch: Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [An Overview of Loss Functions in Deep Learning](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c)
- [Understanding Cross-Entropy Loss](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
