# Image Classification

## Daftar Isi

- [Apa itu Image Classification?](#apa-itu-image-classification)
- [Pipeline Klasifikasi Gambar](#pipeline-klasifikasi-gambar)
- [Dataset Populer](#dataset-populer)
- [Metrik Evaluasi](#metrik-evaluasi)
- [Implementasi dengan PyTorch](#implementasi-dengan-pytorch)
- [Tips Praktis](#tips-praktis)
- [Referensi](#referensi)

---

## Apa itu Image Classification?

Image classification adalah task di mana model diberikan sebuah gambar, dan harus menentukan gambar tersebut termasuk kategori atau kelas mana dari sekumpulan kelas yang sudah didefinisikan.

Contoh sederhana: diberikan foto binatang, model harus mengatakan apakah itu "kucing", "anjing", atau "burung".

Ini adalah task computer vision yang paling dasar, tapi menjadi fondasi untuk task yang lebih kompleks seperti object detection dan segmentation.

### Analogi Sederhananya

Bayangkan seorang ahli botani yang diminta menentukan jenis tanaman hanya dari fotonya. Ia melihat seluruh foto, memperhatikan bentuk daun, warna bunga, struktur batang, lalu memberikan nama spesiesnya. CNN melakukan hal yang sama — memproses gambar secara hierarkis dan menghasilkan label kelas di akhir.

---

## Pipeline Klasifikasi Gambar

Proses lengkap dari data mentah hingga model yang bisa digunakan:

```
1. Data Collection & Labeling
       ↓
2. Preprocessing & Augmentation
       ↓
3. Model Design / Selection
       ↓
4. Training
       ↓
5. Evaluation
       ↓
6. Inference / Deployment
```

### 1. Data Collection & Labeling

Kumpulkan gambar dan beri label kelas untuk setiap gambar. Idealnya data harus:
- Seimbang antar kelas (tidak ada kelas yang dominan berlebihan)
- Beragam (berbagai sudut, cahaya, latar belakang)
- Representatif terhadap kondisi nyata

### 2. Preprocessing

Sebelum gambar dimasukkan ke model, beberapa preprocessing standar perlu dilakukan:

- **Resize**: Semua gambar disamakan ukurannya (misal 224×224 untuk ResNet)
- **Normalisasi**: Nilai pixel di-scale dari [0, 255] ke [0, 1] atau dengan mean/std ImageNet
- **Konversi format**: Dari HWC (height, width, channel) ke CHW (channel, height, width) untuk PyTorch

### 3. Augmentasi Data

Augmentasi secara artifisial memperbesar dataset dengan mentransformasi gambar yang sudah ada. Tujuannya adalah membuat model lebih robust terhadap variasi yang mungkin muncul di data nyata.

| Teknik Augmentasi | Efek |
|---|---|
| Random Horizontal Flip | Model tidak bias terhadap orientasi kiri/kanan |
| Random Crop | Model fokus pada bagian objek, bukan selalu gambar penuh |
| Color Jitter | Model robust terhadap variasi pencahayaan dan warna |
| Random Rotation | Model robust terhadap sedikit kemiringan |
| Gaussian Blur | Model tidak terlalu bergantung pada detail tajam |

---

## Dataset Populer

| Dataset | Kelas | Jumlah Gambar | Ukuran Gambar | Kegunaan |
|---|---|---|---|---|
| **MNIST** | 10 (digit 0-9) | 70,000 | 28×28 (grayscale) | Tutorial, belajar pertama |
| **CIFAR-10** | 10 | 60,000 | 32×32 | Benchmark kecil |
| **CIFAR-100** | 100 | 60,000 | 32×32 | Benchmark lebih sulit |
| **ImageNet** | 1,000 | 1.2 juta | Bervariasi (224×224 umum) | Benchmark standar industri |
| **Flowers-102** | 102 | ~8,000 | Bervariasi | Transfer learning demo |

---

## Metrik Evaluasi

### Accuracy

```
Accuracy = Jumlah prediksi benar / Total prediksi
```

Mudah dipahami, tapi menyesatkan jika dataset tidak seimbang. Contoh: jika 95% data adalah kelas A, model yang selalu prediksi A punya accuracy 95% tapi tidak berguna.

### Confusion Matrix

Menunjukkan distribusi prediksi untuk setiap kombinasi kelas aktual vs prediksi:

```
                Prediksi
              Kucing  Anjing  Burung
Aktual Kucing   45      3       2
       Anjing    2     48       0
       Burung    1      1      48
```

Diagonal = prediksi benar. Off-diagonal = kesalahan klasifikasi.

### Precision, Recall, F1-Score

Metrik yang lebih informatif terutama untuk dataset tidak seimbang:

```
Precision = TP / (TP + FP)   ← dari semua prediksi positif, berapa yang benar?
Recall    = TP / (TP + FN)   ← dari semua yang benar-benar positif, berapa yang ditemukan?
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## Implementasi dengan PyTorch

### Dataset: CIFAR-10 Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt

# ── 1. Data Preprocessing & Augmentation ──────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

# ── 2. Load Dataset ────────────────────────────────────────────────────────────
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=train_transform)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ── 3. Model Architecture ──────────────────────────────────────────────────────
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 32×32 → 16×16
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 16×16 → 8×8
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 8×8 → 4×4
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CIFAR10CNN().to(device)
print(f"Model training di: {device}")

# ── 4. Training Setup ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# ── 5. Training Loop ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
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
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
    
    return total_loss / len(loader), 100. * correct / total

# Training
n_epochs = 50
for epoch in range(1, n_epochs+1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion)
    scheduler.step()
    
    if epoch % 5 == 0:
        print(f"Epoch [{epoch:2d}/{n_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.1f}%")

# ── 6. Inference pada Gambar Baru ──────────────────────────────────────────────
def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Tambah batch dimension
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = probabilities.argmax().item()
    return classes[predicted_class], probabilities[predicted_class].item()

# Ambil satu gambar dari test set
sample_image, sample_label = test_dataset[0]
pred_class, confidence = predict_image(model, sample_image)
print(f"\nGround Truth: {classes[sample_label]}")
print(f"Prediksi:     {pred_class} (confidence: {confidence:.1%})")
```

---

## Tips Praktis

### Batch Normalization

Selalu gunakan Batch Normalization (`nn.BatchNorm2d`) setelah conv layer dan sebelum activation function di arsitektur yang dalam. Ini menstabilkan training dan mempercepat konvergensi.

### Weight Initialization

PyTorch secara default menggunakan He initialization untuk Conv2d, yang sudah optimal untuk ReLU. Jangan ubah tanpa alasan kuat.

### Learning Rate

- Adam: mulai dari `lr=0.001`
- SGD + Momentum: mulai dari `lr=0.01`, biasanya butuh scheduler

### Early Stopping

Monitor validation loss. Jika validation loss mulai naik sementara training loss terus turun, itu tanda overfitting — hentikan training dan ambil checkpoint sebelumnya.

```python
best_val_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(n_epochs):
    # ... training ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping di epoch {epoch}")
            break
```

---

## Referensi

- [PyTorch: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Stanford CS231n: Image Classification](https://cs231n.github.io/classification/)
- [Deep Learning Book - Ian Goodfellow, Chapter 9](https://www.deeplearningbook.org/contents/convnets.html)
- [Torchvision Datasets](https://pytorch.org/vision/stable/datasets.html)
