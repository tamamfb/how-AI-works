# Convolution & Pooling

## Daftar Isi

- [Apa itu Konvolusi?](#apa-itu-konvolusi)
- [Operasi Konvolusi pada Gambar](#operasi-konvolusi-pada-gambar)
- [Stride & Padding](#stride--padding)
- [Feature Map & Multiple Filters](#feature-map--multiple-filters)
- [Pooling Layer](#pooling-layer)
- [Dimensi Output](#dimensi-output)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

---

## Apa itu Konvolusi?

Konvolusi adalah operasi matematis yang menggabungkan dua fungsi untuk menghasilkan fungsi ketiga. Dalam konteks CNN, konvolusi digunakan untuk **mengekstraksi fitur** dari gambar dengan cara menggeser sebuah "kaca pembesar" kecil (filter/kernel) di atas gambar.

### Analogi Sederhananya

Bayangkan kamu punya foto besar dan sebuah lensa kecil berukuran 3x3 cm. Kamu geser lensa itu di atas foto dari sudut kiri atas ke kanan bawah, dan di setiap posisi, lensa itu membuat "catatan" tentang pola yang terlihat di area tersebut. Kumpulan catatan itu adalah hasil konvolusi.

Bedanya, filter CNN tidak hanya melihat — ia **mendeteksi pola spesifik** seperti tepi vertikal, tepi horizontal, tekstur tertentu, dll. Dan model belajar sendiri pola apa yang paling berguna untuk di-detect.

---

## Operasi Konvolusi pada Gambar

Sebuah filter (atau kernel) adalah matriks kecil berisi angka-angka yang dipelajari oleh model. Konvolusi menghitung **dot product** antara filter dengan area gambar yang sedang di-scan.

### Contoh Perhitungan

Input gambar (5×5) dan filter (3×3):

```
Input:          Filter:
1 2 3 0 1       1  0 -1
0 1 2 3 1       1  0 -1
1 0 1 2 0       1  0 -1
0 1 0 1 1
1 0 1 0 1
```

Di posisi pojok kiri atas, konvolusi menghitung:

```
(1×1) + (2×0) + (3×-1) +
(0×1) + (1×0) + (2×-1) +
(1×1) + (0×0) + (1×-1)
= 1 + 0 - 3 + 0 + 0 - 2 + 1 + 0 - 1
= -4
```

Ini adalah satu nilai di output (feature map). Filter lalu digeser ke kanan satu langkah dan proses diulang hingga seluruh gambar ter-scan.

### Apa yang Dideteksi Filter?

Filter yang berbeda mendeteksi fitur yang berbeda:
- Filter tepi vertikal: mendeteksi garis tegak di gambar
- Filter tepi horizontal: mendeteksi garis mendatar
- Filter blur: menghaluskan gambar
- Filter sharpen: mempertajam detail

Dalam CNN, model **belajar sendiri** nilai-nilai filter yang paling berguna untuk task yang diberikan.

---

## Stride & Padding

### Stride

**Stride** adalah seberapa jauh filter bergerak di setiap langkah. Default-nya adalah 1 (geser 1 piksel).

```
Stride = 1 → filter geser satu piksel per langkah (output lebih besar)
Stride = 2 → filter geser dua piksel per langkah (output lebih kecil, informasi lebih padat)
```

Stride yang lebih besar menghasilkan feature map yang lebih kecil, yang bisa mereduksi komputasi dan memori.

### Padding

Masalah: tanpa padding, setiap kali konvolusi diaplikasikan, ukuran gambar mengecil. Setelah banyak layer, gambar bisa jadi sangat kecil.

**Padding** menambahkan "bingkai" nol di sekeliling gambar input sebelum konvolusi dilakukan.

| Tipe Padding | Penjelasan | Efek pada Ukuran |
|---|---|---|
| **Valid** (no padding) | Tidak ada padding, filter hanya di area valid | Output lebih kecil dari input |
| **Same** | Padding ditambahkan agar output = ukuran input | Output sama dengan input |

```
Input (5×5) + Same Padding (1 piksel):

0 0 0 0 0 0 0
0 1 2 3 0 1 0
0 0 1 2 3 1 0
0 1 0 1 2 0 0
0 0 1 0 1 1 0
0 1 0 1 0 1 0
0 0 0 0 0 0 0
```

---

## Feature Map & Multiple Filters

Hasil dari satu konvolusi (satu filter) disebut **feature map** atau activation map. Ia merepresentasikan seberapa kuat fitur yang dideteksi filter itu muncul di setiap lokasi gambar.

Dalam praktek, sebuah convolutional layer menggunakan banyak filter sekaligus:

```
Input: gambar 224×224×3 (tinggi × lebar × channel warna RGB)

Conv Layer dengan 64 filter ukuran 3×3:
→ Output: 224×224×64 (64 feature maps)
```

Setiap dari 64 filter belajar mendeteksi pola yang berbeda. Layer pertama biasanya mendeteksi fitur low-level (tepi, warna), layer-layer berikutnya mendeteksi fitur semakin kompleks (mata, hidung, wajah).

---

## Pooling Layer

Pooling layer bertugas untuk **mereduksi ukuran** feature map (downsampling) setelah konvolusi. Ini membantu:
1. Mengurangi jumlah parameter dan komputasi
2. Memberikan sedikit translational invariance (objek di posisi sedikit berbeda tetap dikenali)

### Max Pooling

Mengambil nilai **maksimum** dari setiap region:

```
Feature Map:       Max Pooling (2×2, stride 2):
1  3  2  4         
5  6  1  2    →    6  4
3  2  7  5         3  7
1  0  3  1
```

Max pooling mengambil nilai terbesar dari setiap blok 2×2. Ini mempertahankan fitur paling menonjol di area tersebut.

### Average Pooling

Mengambil nilai **rata-rata** dari setiap region:

```
1  3         →   (1+3+5+6)/4 = 3.75
5  6
```

Average pooling lebih halus dibanding max pooling, tapi sering kurang efektif untuk klasifikasi gambar.

### Global Average Pooling

Sering digunakan di akhir CNN sebelum fully connected layer. Ia mengambil rata-rata **seluruh feature map** menjadi satu nilai per channel:

```
Feature Map 7×7 → satu nilai (rata-rata semua 49 nilai)
```

Jika ada 512 feature maps, Global Average Pooling menghasilkan vektor dengan 512 nilai.

---

## Dimensi Output

Rumus untuk menghitung dimensi output setelah konvolusi:

```
Output_size = floor((Input_size - Filter_size + 2 × Padding) / Stride) + 1
```

**Contoh:**
- Input: 32×32
- Filter: 5×5
- Padding: 0
- Stride: 1

```
Output = floor((32 - 5 + 2×0) / 1) + 1 = floor(27) + 1 = 28
→ Output: 28×28
```

---

## Implementasi

### Konvolusi Manual (NumPy)

```python
import numpy as np

def conv2d(image, filter_kernel, stride=1, padding=0):
    h_in, w_in = image.shape
    h_f, w_f   = filter_kernel.shape

    # Tambahkan padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')

    h_out = (h_in + 2*padding - h_f) // stride + 1
    w_out = (w_in + 2*padding - w_f) // stride + 1

    output = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            region = image[i*stride:i*stride+h_f, j*stride:j*stride+w_f]
            output[i, j] = np.sum(region * filter_kernel)

    return output

# Contoh: deteksi tepi vertikal
image = np.array([
    [0, 0, 255, 255],
    [0, 0, 255, 255],
    [0, 0, 255, 255],
    [0, 0, 255, 255]
], dtype=float)

edge_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

result = conv2d(image, edge_filter, padding=1)
print(result)
```

### Conv Layer dengan PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Contoh CNN sederhana
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Conv Layer: input 1 channel (grayscale), output 32 filter, kernel 3×3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling: 2×2 max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        self.fc = nn.Linear(64 * 7 * 7, 10)  # untuk input 28×28

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28×28 → 14×14
        x = self.pool(F.relu(self.conv2(x)))  # 14×14 → 7×7
        x = x.view(-1, 64 * 7 * 7)            # Flatten
        x = self.fc(x)
        return x

model = SimpleCNN()

# Test dengan dummy input (batch_size=4, channels=1, height=28, width=28)
dummy_input = torch.randn(4, 1, 28, 28)
output = model(dummy_input)
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")  # (4, 10) — 4 sampel, 10 kelas

# Visualisasi dimensi di setiap layer
x = dummy_input
print(f"\nDimensi di setiap tahap:")
print(f"Input:           {x.shape}")
x = F.relu(model.conv1(x))
print(f"Setelah Conv1:   {x.shape}")
x = model.pool(x)
print(f"Setelah Pool1:   {x.shape}")
x = F.relu(model.conv2(x))
print(f"Setelah Conv2:   {x.shape}")
x = model.pool(x)
print(f"Setelah Pool2:   {x.shape}")
```

---

## Referensi

- [Stanford CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [Deep Learning Book - Ian Goodfellow, Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- [PyTorch: torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [A Guide to Convolutional Arithmetic — Dumoulin & Visin](https://arxiv.org/abs/1603.07285)
