# Arsitektur CNN Populer

## Daftar Isi

- [Evolusi Arsitektur CNN](#evolusi-arsitektur-cnn)
- [LeNet-5 (1998)](#lenet-5-1998)
- [AlexNet (2012)](#alexnet-2012)
- [VGGNet (2014)](#vggnet-2014)
- [ResNet (2015)](#resnet-2015)
- [Inception / GoogLeNet (2014)](#inception--googlenet-2014)
- [EfficientNet (2019)](#efficientnet-2019)
- [Perbandingan Arsitektur](#perbandingan-arsitektur)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

---

## Evolusi Arsitektur CNN

Setiap arsitektur CNN lahir dari upaya untuk memecahkan masalah spesifik yang ada pada arsitektur sebelumnya. Memahami evolusi ini tidak hanya membantumu memilih arsitektur yang tepat, tapi juga melatih intuisi tentang kenapa desain tertentu bekerja dengan baik.

```
Tahun:  1998      2012       2014      2014       2015      2019
        LeNet → AlexNet → VGGNet → GoogLeNet → ResNet → EfficientNet
        (7 layer) (8 layer) (19 layer) (22 layer) (152 layer) (B7: 66M params)
```

---

## LeNet-5 (1998)

**Paper:** "Gradient-Based Learning Applied to Document Recognition" — Yann LeCun et al.

LeNet-5 adalah CNN pertama yang berhasil diaplikasikan secara komersial, digunakan untuk membaca tulisan tangan pada cek bank oleh AT&T.

### Arsitektur

```
Input (32×32×1)
    ↓
Conv1 (6 filter, 5×5) → 28×28×6
    ↓
AvgPool (2×2) → 14×14×6
    ↓
Conv2 (16 filter, 5×5) → 10×10×16
    ↓
AvgPool (2×2) → 5×5×16
    ↓
Flatten → 400
    ↓
FC1 (120) → FC2 (84) → FC3 (10)
    ↓
Output (10 kelas)
```

**Inovasi utama:** Membuktikan bahwa parameter sharing dan local connectivity bisa bekerja sangat efektif untuk pengenalan gambar.

---

## AlexNet (2012)

**Paper:** "ImageNet Classification with Deep Convolutional Neural Networks" — Krizhevsky, Sutskever, Hinton

AlexNet adalah titik balik sejarah deep learning. Pada kompetisi ImageNet 2012, AlexNet menang dengan top-5 error rate 15.3% — jauh lebih baik dari runner-up yang menggunakan metode tradisional (26.2%).

### Inovasi AlexNet

1. **Menggunakan ReLU** — bukan Tanh/Sigmoid. Lebih cepat dan tidak mengalami vanishing gradient seberat Sigmoid.
2. **Pelatihan di GPU** — AlexNet adalah arsitektur besar pertama yang sengaja didesain untuk dilatih di GPU (dua GPU diparalelkan).
3. **Dropout** — regularisasi untuk mencegah overfitting di fully connected layer.
4. **Data Augmentation** — random cropping dan flipping untuk memperbesar dataset secara efektif.
5. **Local Response Normalization (LRN)** — normalisasi yang kemudian jarang digunakan karena digantikan Batch Normalization.

### Arsitektur (disederhanakan)

```
Input (227×227×3)
    ↓
Conv1 (96 filter, 11×11, stride 4) → 55×55×96 + MaxPool → 27×27×96
    ↓
Conv2 (256 filter, 5×5, same) → 27×27×256 + MaxPool → 13×13×256
    ↓
Conv3 (384 filter, 3×3, same) → 13×13×384
Conv4 (384 filter, 3×3, same) → 13×13×384
Conv5 (256 filter, 3×3, same) → 13×13×256 + MaxPool → 6×6×256
    ↓
Flatten → 9216
FC1 (4096) + Dropout → FC2 (4096) + Dropout → FC3 (1000)
    ↓
Output Softmax (1000 kelas)
```

Total parameter: ~60 juta

---

## VGGNet (2014)

**Paper:** "Very Deep Convolutional Networks for Large-Scale Image Recognition" — Simonyan & Zisserman (Oxford)

Gagasan utama VGG: **keep it simple and go deeper**. Gunakan hanya filter 3×3 dengan stride 1 secara konsisten di seluruh jaringan.

### Mengapa 3×3 Saja?

Dua filter 3×3 berturut-turut memiliki **receptive field** yang sama dengan satu filter 5×5, tapi:
- Lebih sedikit parameter: 2×(3×3) = 18 vs 1×(5×5) = 25
- Lebih banyak non-linearitas (dua activation function vs satu)

Tiga filter 3×3 = receptive field 7×7, dengan parameter jauh lebih sedikit.

### Arsitektur VGG-16 (16 weight layer)

```
Input (224×224×3)
Block 1: Conv 3×3 ×2 (64 filter) + MaxPool → 112×112×64
Block 2: Conv 3×3 ×2 (128 filter) + MaxPool → 56×56×128
Block 3: Conv 3×3 ×3 (256 filter) + MaxPool → 28×28×256
Block 4: Conv 3×3 ×3 (512 filter) + MaxPool → 14×14×512
Block 5: Conv 3×3 ×3 (512 filter) + MaxPool → 7×7×512
Flatten → FC (4096) → FC (4096) → FC (1000) → Softmax
```

Total parameter: ~138 juta

**Kelemahan:** Sangat berat — 138 juta parameter, memori besar untuk inference.

---

## ResNet (2015)

**Paper:** "Deep Residual Learning for Image Recognition" — He et al. (Microsoft Research)

ResNet memecahkan masalah **degradasi** — fenomena di mana menambah lebih banyak layer pada jaringan yang sangat dalam justru membuat akurasi training turun. Ini bukan overfitting; modelnya bahkan lebih buruk di training set.

### Masalah: Vanishing Gradient di Jaringan Dalam

Saat jaringan sangat dalam (50+ layer), gradient menjadi sangat kecil saat backprop melewati banyak layer. Akibatnya, layer-layer awal hampir tidak mendapat sinyal untuk belajar.

### Solusi: Residual Connection (Skip Connection)

ResNet menambahkan "jalan pintas" yang melewatkan input langsung ke output beberapa layer di depan:

```
       Input x
       │    │
       │    └──────────────┐
       ↓                   │
   [Conv 3×3]              │  (skip connection)
   [BN + ReLU]             │
   [Conv 3×3]              │
   [BN]                    │
       │                   │
       └────────(+)←───────┘
                │
           [ReLU]
               ↓
            Output
```

Secara matematis: `F(x) + x` — model hanya perlu belajar **residual** (selisih) dari transformasi yang diinginkan, bukan transformasi penuh. Ini jauh lebih mudah untuk dioptimasi.

Bahkan jika weight layer di tengah mendekati nol, identity shortcut memastikan gradient tetap bisa mengalir balik.

### Varian ResNet

| Arsitektur | Jumlah Layer | Top-5 Error (ImageNet) | Parameter |
|---|---|---|---|
| ResNet-18  | 18  | ~10.9% | 11.7M |
| ResNet-34  | 34  | ~10.0% | 21.8M |
| ResNet-50  | 50  | ~7.0%  | 25.6M |
| ResNet-101 | 101 | ~6.0%  | 44.6M |
| ResNet-152 | 152 | ~5.7%  | 60.4M |

---

## Inception / GoogLeNet (2014)

**Paper:** "Going Deeper with Convolutions" — Szegedy et al. (Google)

Pertanyaan yang dijawab Inception: **daripada bertanya "harus pakai filter berukuran berapa?", kenapa tidak pakai semua sekaligus?**

### Inception Module

```
Input
 │───────────────────────────────────────────────────┐
 │           │              │                         │
 ↓           ↓              ↓                         ↓
Conv 1×1   Conv 1×1      Conv 1×1              MaxPool 3×3
           ↓              ↓                         ↓
         Conv 3×3       Conv 5×5               Conv 1×1
           │              │                         │
           └──────────────┴─────────────────────────┘
                             Concatenate
```

Setiap inception module menggunakan filter 1×1, 3×3, dan 5×5 secara paralel, lalu menggabungkan hasilnya. Model bisa belajar sendiri fitur mana yang paling berguna.

Filter 1×1 juga digunakan untuk **dimension reduction** — mengurangi jumlah channel sebelum filter yang lebih besar, sehingga komputasi jauh lebih efisien.

GoogLeNet (22 layer) memiliki ~5 juta parameter — jauh lebih efisien dari AlexNet (60M) atau VGGNet (138M) dengan akurasi yang lebih baik.

---

## EfficientNet (2019)

**Paper:** "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" — Tan & Le (Google Brain)

EfficientNet mengatasi pertanyaan klasik: **bagaimana cara terbaik untuk scale up sebuah CNN?** Ada tiga dimensi yang bisa di-scale:
- **Depth** (lebih banyak layer)
- **Width** (lebih banyak filter per layer)
- **Resolution** (input gambar lebih besar)

Biasanya orang hanya scale satu dimensi. EfficientNet memperkenalkan **compound scaling** — scale ketiganya secara bersamaan dengan rasio yang diseimbangkan menggunakan Neural Architecture Search (NAS).

### Varian EfficientNet

| Model | Top-1 Acc (ImageNet) | Parameter | FLOPs |
|---|---|---|---|
| EfficientNet-B0 | 77.1% | 5.3M  | 0.39B |
| EfficientNet-B3 | 81.6% | 12M   | 1.8B  |
| EfficientNet-B7 | 84.3% | 66M   | 37B   |

EfficientNet-B7 mencapai akurasi terbaik ImageNet saat itu dengan parameter **8× lebih sedikit** dan **6× lebih cepat** dari ResNet-152 dengan akurasi setara.

---

## Perbandingan Arsitektur

| Arsitektur | Tahun | Top-5 Error (ImageNet) | Parameter | Inovasi Utama |
|---|---|---|---|---|
| LeNet-5    | 1998 | N/A (MNIST)   | ~60K  | CNN pertama yang berhasil |
| AlexNet    | 2012 | 15.3%         | 60M   | ReLU, GPU training, Dropout |
| VGGNet-16  | 2014 | 7.3%          | 138M  | Filter 3×3 konsisten, sangat dalam |
| GoogLeNet  | 2014 | 6.7%          | 5M    | Inception module, efisien |
| ResNet-50  | 2015 | 5.3%          | 25.6M | Residual / skip connections |
| EfficientNet-B7 | 2019 | 1.8%    | 66M   | Compound scaling |

---

## Implementasi

### Menggunakan Model Pre-trained dari torchvision

```python
import torch
import torchvision.models as models

# Load berbagai arsitektur (pre-trained di ImageNet)
lenet     = models.squeezenet1_0(pretrained=False)   # LeNet-like
alexnet   = models.alexnet(pretrained=True)
vgg16     = models.vgg16(pretrained=True)
resnet50  = models.resnet50(pretrained=True)
inception = models.inception_v3(pretrained=True)

# Lihat arsitektur ResNet-50
print(models.resnet50())
```

### Membangun ResNet Block dari Scratch

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projection shortcut jika dimensi berubah
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)  # Skip connection — inti dari ResNet
        out = self.relu(out)
        return out

# Test residual block
block = ResidualBlock(in_channels=64, out_channels=64)
x = torch.randn(1, 64, 56, 56)
out = block(x)
print(f"Input: {x.shape} → Output: {out.shape}")  # Shape sama karena stride=1
```

---

## Referensi

- [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://www.image-net.org/challenges/LSVRC/)
- [LeNet-5 — LeCun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [AlexNet — Krizhevsky et al., 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [VGGNet — Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)
- [ResNet — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [GoogLeNet/Inception — Szegedy et al., 2014](https://arxiv.org/abs/1409.4842)
- [EfficientNet — Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- [torchvision.models Documentation](https://pytorch.org/vision/stable/models.html)
