# Transfer Learning

## Daftar Isi

- [Apa itu Transfer Learning?](#apa-itu-transfer-learning)
- [Mengapa Transfer Learning Penting?](#mengapa-transfer-learning-penting)
- [Dua Strategi Transfer Learning](#dua-strategi-transfer-learning)
  - [Feature Extraction](#feature-extraction)
  - [Fine-Tuning](#fine-tuning)
- [Model Pre-trained Populer](#model-pre-trained-populer)
- [Kapan Menggunakan Strategi Mana?](#kapan-menggunakan-strategi-mana)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

---

## Apa itu Transfer Learning?

Transfer learning adalah teknik di mana kita menggunakan **pengetahuan yang sudah dipelajari model dari satu task (source)** untuk membantu mempelajari task yang berbeda (target).

Daripada melatih model dari nol, kita mulai dari model yang sudah dilatih di dataset besar dan meng-adaptasinya untuk kebutuhan kita.

### Analogi Sederhananya

Seorang dokter yang baru lulus dan ingin menjadi dokter bedah tidak mulai belajar dari nol tentang anatomi manusia, biologi, dan kimia. Ia membawa seluruh pengetahuan medis yang sudah dipelajari bertahun-tahun, lalu hanya fokus mempelajari teknik pembedahan secara khusus.

Transfer learning bekerja persis seperti itu — pengetahuan umum (fitur visual dasar: tepi, tekstur, bentuk) yang sudah dipelajari dari ImageNet ditransfer ke task spesifik (misalnya deteksi penyakit pada X-ray).

---

## Mengapa Transfer Learning Penting?

### Masalah tanpa Transfer Learning

Melatih CNN besar dari nol membutuhkan:
- Dataset yang sangat besar (ratusan ribu hingga jutaan gambar)
- Waktu training yang panjang (hari hingga minggu)
- Hardware mahal (GPU cluster)

Ini tidak realistis untuk kebanyakan proyek di dunia nyata, di mana dataset mungkin hanya ribuan gambar dan komputasi terbatas.

### Solusi Transfer Learning

Dengan model pre-trained seperti ResNet yang sudah dilatih di ImageNet (1.2 juta gambar, 1000 kelas):

- Layer-layer awal sudah belajar mendeteksi fitur universal: tepi, sudut, tekstur, gradien warna
- Layer-layer tengah sudah belajar pola lebih kompleks: mata, roda, daun
- Hanya layer-layer akhir yang perlu disesuaikan dengan task baru

Hasilnya: training jauh lebih cepat, akurasi lebih tinggi dengan data lebih sedikit.

---

## Dua Strategi Transfer Learning

### Feature Extraction

Semua layer pre-trained di-**freeze** (bobotnya tidak diubah). Kita hanya menambahkan dan melatih **classifier baru** di atas fitur yang dihasilkan model.

```
Pre-trained Model (FROZEN)                  New Head (TRAINABLE)
┌─────────────────────────┐              ┌──────────────────────┐
│  Conv Layers            │              │  Global Avg Pooling  │
│  (belajar fitur umum)   │──── output ──│  FC Layer            │
│  [TIDAK diperbarui]     │              │  Softmax (N kelas)   │
└─────────────────────────┘              └──────────────────────┘
```

**Kapan digunakan:**
- Dataset target sangat kecil (< 1000 gambar per kelas)
- Dataset target mirip dengan dataset source (misal: keduanya foto objek sehari-hari)
- Sumber daya komputasi terbatas

---

### Fine-Tuning

Sebagian atau semua layer pre-trained **di-unfreeze** dan dilatih ulang dengan learning rate yang sangat kecil, bersamaan dengan classifier baru.

```
Pre-trained Model (SEBAGIAN/SELURUHNYA TRAINABLE)   New Head (TRAINABLE)
┌────────────────────────────────┐               ┌──────────────────────┐
│  Layer awal: FROZEN            │               │  Global Avg Pooling  │
│  (fitur sangat umum)           │──── output ───│  FC Layer            │
│                                │               │  Softmax (N kelas)   │
│  Layer akhir: TRAINABLE        │               └──────────────────────┘
│  (disesuaikan ke domain baru)  │
└────────────────────────────────┘
```

**Kapan digunakan:**
- Dataset target cukup besar (> 1000 gambar per kelas)
- Dataset target berbeda signifikan dari ImageNet (misal: citra medis, satelit)

**Penting:** Gunakan learning rate yang sangat kecil saat fine-tuning (10x hingga 100x lebih kecil dari training normal) agar "pengetahuan" yang sudah ada tidak rusak.

---

## Model Pre-trained Populer

| Model | Dataset | Parameter | Top-1 Acc | Cocok untuk |
|---|---|---|---|---|
| **ResNet-50** | ImageNet | 25.6M | 76.1% | General purpose, stable |
| **ResNet-101** | ImageNet | 44.6M | 77.4% | Akurasi lebih tinggi |
| **VGG-16** | ImageNet | 138M | 71.6% | Simple, banyak tutorial |
| **EfficientNet-B0** | ImageNet | 5.3M | 77.1% | Efisien, mobile-friendly |
| **EfficientNet-B4** | ImageNet | 19M | 82.9% | High accuracy |
| **MobileNetV2** | ImageNet | 3.4M | 72.0% | Mobile/embedded deployment |
| **DenseNet-121** | ImageNet | 8M | 74.9% | Medical imaging |
| **ViT-B/16** | ImageNet-21k | 86M | 81.8% | Vision Transformer |

---

## Kapan Menggunakan Strategi Mana?

Panduan memilih strategi berdasarkan ukuran dataset dan kemiripan domain:

```
                    Dataset Kecil        Dataset Besar
                  (< ~1000/kelas)      (> ~1000/kelas)
                 ┌──────────────────┬──────────────────┐
Domain           │ Feature          │ Fine-tune semua  │
Mirip            │ Extraction       │ layer dengan     │
(foto umum)      │ saja             │ lr kecil         │
                 ├──────────────────┼──────────────────┤
Domain           │ Fine-tune layer  │ Fine-tune semua  │
Berbeda          │ akhir + head     │ layer, mungkin   │
(medis, satelit) │ baru             │ perlu dari nol   │
                 └──────────────────┴──────────────────┘
```

---

## Implementasi

### Feature Extraction dengan ResNet-50

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ── 1. Load Pre-trained Model ──────────────────────────────────────────────────
model = models.resnet50(pretrained=True)

# Freeze semua parameter
for param in model.parameters():
    param.requires_grad = False

# Ganti layer classifier terakhir (fully connected)
# ResNet-50 menggunakan model.fc sebagai layer terakhir
n_features = model.fc.in_features   # = 2048
n_classes  = 5                       # Jumlah kelas dataset kita

model.fc = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, n_classes)
)
# Hanya model.fc yang trainable karena parameter lain di-freeze

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Verifikasi parameter yang trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")

# ── 2. Preprocessing ───────────────────────────────────────────────────────────
# Gunakan mean/std ImageNet karena model di-pretrain di sana
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# ── 3. Training ────────────────────────────────────────────────────────────────
# Hanya optimalkan parameter yang trainable (model.fc)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    
    return total_loss / len(loader), correct / total
```

### Fine-Tuning Bertahap (Gradual Unfreezing)

Teknik yang lebih canggih: buka layer dari akhir ke awal secara bertahap.

```python
import torchvision.models as models
import torch.optim as optim

model = models.resnet50(pretrained=True)

# Ganti head classifier
model.fc = nn.Linear(model.fc.in_features, n_classes)

# Tahap 1: Hanya latih head (epoch 1-5)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
# ... training 5 epoch ...

# Tahap 2: Buka layer 4 (epoch 6-15)
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {'params': model.fc.parameters(),     'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # lr lebih kecil
])
# ... training 10 epoch ...

# Tahap 3: Buka semua layer (epoch 16+)
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])
```

### Contoh End-to-End: Klasifikasi Dataset Kustom

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def build_model(n_classes, pretrained=True, freeze_backbone=True):
    model = models.resnet50(pretrained=pretrained)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, n_classes)
    )
    return model

# Struktur direktori dataset:
# dataset/
#   train/
#     kelas_1/ ← berisi gambar-gambar kelas 1
#     kelas_2/
#   val/
#     kelas_1/
#     kelas_2/

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_dataset = ImageFolder('dataset/train', transform=transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
]))

val_dataset = ImageFolder('dataset/val', transform=transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
]))

n_classes = len(train_dataset.classes)
print(f"Kelas: {train_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = build_model(n_classes).to(device)

optimizer = nn.CrossEntropyLoss()
optimizer_fn = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
criterion = nn.CrossEntropyLoss()

# Training (5 epoch cukup untuk feature extraction dengan model pre-trained)
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_fn.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_fn.step()
    
    # Evaluasi
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    
    print(f"Epoch {epoch+1} | Val Accuracy: {100*correct/total:.1f}%")

# Simpan model
torch.save(model.state_dict(), 'my_model.pth')
```

---

## Referensi

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CS231n: Transfer Learning](https://cs231n.github.io/transfer-learning/)
- [A Survey on Transfer Learning — Pan & Yang, 2010](https://ieeexplore.ieee.org/document/5288526)
- [How transferable are features in deep neural networks? — Yosinski et al., 2014](https://arxiv.org/abs/1411.1792)
- [torchvision.models](https://pytorch.org/vision/stable/models.html)
