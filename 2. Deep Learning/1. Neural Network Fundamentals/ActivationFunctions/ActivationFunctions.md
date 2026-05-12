# Activation Functions

## Daftar Isi

- [Apa itu Activation Function?](#apa-itu-activation-function)
- [Kenapa Perlu Non-Linearitas?](#kenapa-perlu-non-linearitas)
- [Jenis-Jenis Activation Function](#jenis-jenis-activation-function)
  - [Sigmoid](#sigmoid)
  - [Tanh](#tanh)
  - [ReLU](#relu)
  - [Leaky ReLU](#leaky-relu)
  - [ELU](#elu)
  - [Softmax](#softmax)
- [Kapan Menggunakan Apa?](#kapan-menggunakan-apa)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

---

## Apa itu Activation Function?

Setelah neuron menghitung weighted sum (`z = Σwᵢxᵢ + b`), hasilnya dilewatkan melalui sebuah **activation function** — fungsi yang menentukan apakah dan seberapa kuat neuron tersebut "aktif".

Tanpa activation function, neural network — seberapapun dalamnya — hanyalah sekumpulan perkalian matriks yang bisa disederhanakan menjadi satu persamaan linear. Itu berarti model tidak akan bisa mempelajari pola yang kompleks.

### Analogi Sederhananya

Bayangkan sebuah lampu dimmer. Input adalah putaran kenop, dan activation function menentukan seberapa terang lampunya. Tanpa dimmer, lampu hanya bisa menyala penuh atau mati — tidak ada gradasi. Dengan activation function yang tepat, neuron bisa mengekspresikan berbagai tingkat "aktivasi".

---

## Kenapa Perlu Non-Linearitas?

Jika semua activation function adalah linear (misalnya `f(z) = z`), maka:

```
Layer 1: h₁ = W₁x + b₁
Layer 2: h₂ = W₂h₁ + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
```

Hasilnya tetap persamaan linear — tidak peduli berapa banyak layer yang ditambahkan. Dengan fungsi aktivasi non-linear, setiap layer bisa mempelajari transformasi yang lebih kompleks dan model bisa merepresentasikan hubungan yang tidak linear antara input dan output.

---

## Jenis-Jenis Activation Function

### Sigmoid

**Formula:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Output range:** (0, 1)

Sigmoid mengubah semua nilai input menjadi nilai antara 0 dan 1. Cocok untuk **output layer** pada masalah klasifikasi biner karena outputnya bisa diinterpretasikan sebagai probabilitas.

**Kelemahan:**
- **Vanishing gradient**: untuk nilai z yang sangat besar atau sangat kecil, gradiennya mendekati 0. Ini membuat training sangat lambat di layer-layer awal.
- Output tidak zero-centered (rata-rata output bukan di 0), yang bisa memperlambat konvergensi.

```
Output
1.0 |          ________
    |        /
0.5 |      /
    |    /
0.0 |__/
    └───────────────── Input z
     -5   0    5
```

---

### Tanh

**Formula:**
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

**Output range:** (-1, 1)

Tanh mirip dengan Sigmoid tapi output-nya zero-centered, sehingga konvergensi sedikit lebih cepat. Tapi masalah vanishing gradient tetap ada.

```
Output
 1.0 |          ________
     |        /
 0.0 |──────/──────────── Input z
     |    /
-1.0 |__/
     └───────────────────
      -5   0    5
```

---

### ReLU

**Formula:**
```
ReLU(z) = max(0, z)
```

**Output range:** [0, +∞)

ReLU (Rectified Linear Unit) adalah activation function paling populer untuk **hidden layer** saat ini. Sangat sederhana: jika input negatif, output 0; jika positif, output sama dengan input.

**Keunggulan:**
- Komputasi sangat cepat
- Tidak ada vanishing gradient untuk nilai positif
- Mendorong **sparse activation** (banyak neuron bernilai 0) yang efisien

**Kelemahan:**
- **Dying ReLU**: neuron yang selalu menerima input negatif akan selalu output 0 dan berhenti belajar sama sekali

```
Output
  5 |            /
    |          /
  0 |________/
    └───────────── Input z
     -5   0   5
```

---

### Leaky ReLU

**Formula:**
```
LeakyReLU(z) = max(αz, z)  di mana α biasanya = 0.01
```

**Output range:** (-∞, +∞)

Leaky ReLU adalah perbaikan dari ReLU untuk mengatasi masalah dying ReLU. Untuk input negatif, bukannya output 0, outputnya adalah nilai kecil negatif (`αz`). Ini memastikan neuron tetap bisa menerima gradient walaupun inputnya negatif.

---

### ELU

**Formula:**
```
ELU(z) = z               jika z > 0
ELU(z) = α(e^z - 1)      jika z ≤ 0
```

ELU (Exponential Linear Unit) memiliki output negatif yang smooth untuk input negatif, sehingga rata-rata output mendekati nol dan konvergensi lebih stabil. Tapi komputasinya lebih berat karena melibatkan fungsi eksponensial.

---

### Softmax

**Formula:**
```
Softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
```

**Output range:** (0, 1), dengan semua output berjumlah 1

Softmax digunakan khusus di **output layer** untuk masalah **klasifikasi multi-kelas**. Ia mengubah sekumpulan nilai menjadi distribusi probabilitas — setiap output merepresentasikan probabilitas kelas tersebut.

Contoh: jika model mengklasifikasikan gambar ke tiga kelas (kucing, anjing, burung), Softmax akan menghasilkan output seperti `[0.7, 0.2, 0.1]` — 70% kemungkinan kucing, 20% anjing, 10% burung.

---

## Kapan Menggunakan Apa?

| Activation Function | Digunakan Di | Keterangan |
|---|---|---|
| **ReLU** | Hidden layer (default) | Cepat, efektif untuk kebanyakan kasus |
| **Leaky ReLU / ELU** | Hidden layer | Alternatif ReLU jika masalah dying neuron |
| **Sigmoid** | Output layer — klasifikasi biner | Output sebagai probabilitas (0–1) |
| **Softmax** | Output layer — klasifikasi multi-kelas | Output sebagai distribusi probabilitas |
| **Tanh** | Hidden layer (RNN/LSTM) | Lebih baik dari Sigmoid untuk layer tersembunyi |
| **Linear / None** | Output layer — regresi | Nilai output tidak dibatasi |

---

## Implementasi

### Semua Activation Function dengan PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Buat tensor input
x = torch.linspace(-5, 5, 100)

# Hitung output setiap activation function
sigmoid_out   = torch.sigmoid(x)
tanh_out      = torch.tanh(x)
relu_out      = F.relu(x)
leaky_relu_out = F.leaky_relu(x, negative_slope=0.1)
elu_out       = F.elu(x)
softmax_out   = F.softmax(x, dim=0)

# Visualisasi
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
functions = [
    (sigmoid_out,    "Sigmoid",    "blue"),
    (tanh_out,       "Tanh",       "green"),
    (relu_out,       "ReLU",       "red"),
    (leaky_relu_out, "Leaky ReLU", "orange"),
    (elu_out,        "ELU",        "purple"),
    (softmax_out,    "Softmax",    "brown"),
]

for ax, (output, name, color) in zip(axes.flatten(), functions):
    ax.plot(x.numpy(), output.numpy(), color=color, linewidth=2)
    ax.set_title(name)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("activation_functions.png", dpi=100)
plt.show()
```

### Penggunaan dalam Neural Network

```python
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)   # Input ke hidden 1
        self.fc2 = nn.Linear(256, 128)   # Hidden 1 ke hidden 2
        self.fc3 = nn.Linear(128, 10)    # Hidden 2 ke output (10 kelas)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))    # Hidden layer pakai ReLU
        x = self.relu(self.fc2(x))    # Hidden layer pakai ReLU
        x = self.softmax(self.fc3(x)) # Output layer pakai Softmax
        return x
```

---

## Referensi

- [Deep Learning Book - Ian Goodfellow, Section 6.3: Hidden Units](https://www.deeplearningbook.org/contents/mlp.html)
- [Understanding the Vanishing Gradient Problem](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)
- [PyTorch: torch.nn.functional — Activation Functions](https://pytorch.org/docs/stable/nn.functional.html)
- [Practical Recommendations for Gradient-Based Training — LeCun et al.](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
