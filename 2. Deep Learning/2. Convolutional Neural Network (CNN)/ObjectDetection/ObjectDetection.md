# Object Detection

## Daftar Isi

- [Perbedaan Classification vs Detection](#perbedaan-classification-vs-detection)
- [Konsep Dasar Object Detection](#konsep-dasar-object-detection)
- [Pendekatan Two-Stage: R-CNN Family](#pendekatan-two-stage-r-cnn-family)
- [Pendekatan One-Stage: YOLO](#pendekatan-one-stage-yolo)
- [Pendekatan One-Stage: SSD](#pendekatan-one-stage-ssd)
- [Perbandingan Algoritma](#perbandingan-algoritma)
- [Metrik Evaluasi](#metrik-evaluasi)
- [Implementasi](#implementasi)
- [Referensi](#referensi)

---

## Perbedaan Classification vs Detection

| | Image Classification | Object Detection |
|---|---|---|
| **Tugas** | "Gambar ini isinya apa?" | "Objek apa yang ada di gambar ini, dan di mana?" |
| **Output** | Label kelas | Label kelas + bounding box (x, y, w, h) |
| **Tantangan** | Memahami isi gambar secara keseluruhan | Menemukan dan melokalisasi multiple objek |
| **Contoh** | "Ini kucing" | "Ada kucing di koordinat [50,30,200,180] dan anjing di [300,50,450,280]" |

Object detection adalah task yang lebih kompleks karena model harus menjawab dua pertanyaan sekaligus: **apa** (klasifikasi) dan **di mana** (lokalisasi).

---

## Konsep Dasar Object Detection

### Bounding Box

Setiap objek yang dideteksi diwakili oleh **bounding box** — kotak persegi panjang yang mengelilingi objek. Format yang umum digunakan:

```
(x_min, y_min, x_max, y_max)  ← koordinat sudut
atau
(x_center, y_center, width, height)  ← format YOLO
```

### Anchor Box

Anchor box (atau default box) adalah kotak-kotak referensi dengan berbagai ukuran dan rasio aspek yang sudah didefinisikan. Model tidak memprediksi koordinat bounding box dari nol, melainkan memprediksi **offset** (perbedaan) dari anchor box terhadap objek aktual. Ini membuat training lebih stabil.

### Confidence Score

Setiap prediksi bounding box memiliki **confidence score** — seberapa yakin model bahwa ada objek di kotak itu DAN seberapa akurat posisinya.

### Non-Maximum Suppression (NMS)

Model sering menghasilkan banyak bounding box yang overlapping untuk objek yang sama. NMS memilih satu yang terbaik:

1. Urutkan semua bounding box berdasarkan confidence score
2. Pilih yang tertinggi
3. Hapus semua box lain yang overlap terlalu banyak (IoU > threshold) dengan box terpilih
4. Ulangi sampai tidak ada box tersisa

### Intersection over Union (IoU)

IoU mengukur seberapa besar tumpang tindih antara bounding box yang diprediksi dengan ground truth:

```
IoU = Area(Intersection) / Area(Union)

IoU = 1.0  → prediksi sempurna
IoU = 0.5  → threshold umum untuk "deteksi berhasil"
IoU = 0.0  → tidak ada tumpang tindih
```

---

## Pendekatan Two-Stage: R-CNN Family

Algoritma two-stage memisahkan proses menjadi dua tahap: (1) proposal region, (2) klasifikasi + perbaikan lokasi.

### R-CNN (2013)

**Cara kerja:**
1. **Selective Search** menghasilkan ~2000 region proposals dari gambar
2. Setiap region di-crop dan di-resize ke ukuran tetap
3. CNN mengekstraksi fitur dari setiap region
4. SVM mengklasifikasi fitur tersebut
5. Linear regression memperbaiki koordinat bounding box

**Masalah:** Sangat lambat — CNN dijalankan 2000 kali per gambar, masing-masing terpisah.

### Fast R-CNN (2015)

Solusi untuk lambatnya R-CNN: jalankan CNN **sekali** untuk seluruh gambar, bukan untuk setiap proposal.

**Cara kerja:**
1. Jalankan CNN sekali → dapatkan feature map seluruh gambar
2. Project region proposals ke feature map
3. **RoI Pooling** mengekstraksi fitur fixed-size dari setiap region di feature map
4. Fully connected layer mengklasifikasi dan memperbaiki bounding box secara bersamaan

Hasilnya ~10× lebih cepat dari R-CNN.

### Faster R-CNN (2015)

Bottleneck Fast R-CNN adalah selective search yang masih dilakukan di CPU. Faster R-CNN menggantikannya dengan **Region Proposal Network (RPN)** yang dijalankan di GPU.

**Cara kerja:**
1. CNN backbone (VGG, ResNet) mengekstraksi feature map
2. **RPN** berjalan di atas feature map, mengusulkan region yang kemungkinan mengandung objek
3. RoI Pooling mengekstraksi fitur dari region proposals
4. Fully connected head mengklasifikasi dan merefinement bounding box

Faster R-CNN adalah arsitektur two-stage paling berpengaruh dan menjadi baseline untuk banyak penelitian berikutnya.

---

## Pendekatan One-Stage: YOLO

**Paper:** "You Only Look Once: Unified, Real-Time Object Detection" — Redmon et al. (2016)

Ide utama YOLO: mengapa harus dua tahap? Jadikan detection sebagai satu masalah regresi tunggal — langsung dari piksel gambar ke bounding box dan probabilitas kelas.

### Cara Kerja YOLO

1. Gambar dibagi menjadi grid S×S (misalnya 7×7)
2. Setiap sel grid bertanggung jawab mendeteksi objek yang pusat (center)-nya jatuh di sel tersebut
3. Setiap sel memprediksi B bounding box, masing-masing dengan 5 nilai: (x, y, w, h, confidence)
4. Setiap sel juga memprediksi C probabilitas kelas

Output akhir: tensor berukuran S × S × (B×5 + C)

```
Contoh (S=7, B=2, C=20 kelas PASCAL VOC):
Output: 7 × 7 × (2×5 + 20) = 7 × 7 × 30
```

### Keunggulan YOLO

- **Sangat cepat** — bisa mencapai 45-155 FPS (YOLOv2), cocok untuk real-time
- Melihat seluruh gambar sekaligus — lebih sedikit false positive pada latar belakang
- Generalisasi baik — lebih robust pada domain baru

### Kelemahan YOLO (versi awal)

- Kesulitan mendeteksi objek kecil
- Kesulitan mendeteksi objek yang berkelompok dalam satu sel grid

### Evolusi YOLO

| Versi | Tahun | Inovasi Utama |
|---|---|---|
| YOLOv1 | 2016 | Konsep dasar one-shot detection |
| YOLOv2 | 2017 | Anchor boxes, Batch Normalization, multi-scale training |
| YOLOv3 | 2018 | Deteksi multi-skala (FPN-like), Darknet-53 backbone |
| YOLOv4 | 2020 | Bag of Freebies & Specials, CSP backbone |
| YOLOv5 | 2020 | PyTorch native, sangat mudah digunakan |
| YOLOv8 | 2023 | Arsitektur baru, anchor-free, lebih akurat dan cepat |

---

## Pendekatan One-Stage: SSD

**Paper:** "SSD: Single Shot MultiBox Detector" — Liu et al. (2016)

SSD mendeteksi objek di **multiple scales** secara bersamaan — menggunakan feature map dari berbagai kedalaman jaringan.

### Cara Kerja SSD

Feature map yang lebih besar (dari layer awal) cocok untuk mendeteksi objek kecil karena resolusinya masih tinggi. Feature map yang lebih kecil (dari layer akhir) cocok untuk mendeteksi objek besar karena receptive field-nya luas.

SSD memanfaatkan kedua jenis feature map:

```
Input (300×300)
    ↓ VGG16 backbone
Feature Map 38×38  ← deteksi objek kecil
    ↓
Feature Map 19×19
    ↓
Feature Map 10×10
    ↓
Feature Map  5×5
    ↓
Feature Map  3×3
    ↓
Feature Map  1×1   ← deteksi objek besar

Semua feature maps → Prediksi bounding box + kelas → NMS → Hasil akhir
```

---

## Perbandingan Algoritma

| Model | Kecepatan (FPS) | mAP (COCO) | Keunggulan | Kelemahan |
|---|---|---|---|---|
| R-CNN | 0.05 | - | Akurasi tinggi | Sangat lambat |
| Fast R-CNN | 7 | - | Lebih cepat dari R-CNN | Masih dua tahap |
| Faster R-CNN | 17 | ~37% | Akurasi tinggi, lebih cepat | Tidak real-time |
| SSD300 | 46 | ~25% | Cepat, multi-scale | Akurasi objek kecil rendah |
| YOLOv3 | 45-65 | ~33% | Keseimbangan speed-accuracy | - |
| YOLOv8n | 160+ | ~37% | State-of-the-art, mudah dipakai | - |

*FPS dan mAP bergantung pada hardware dan konfigurasi*

---

## Metrik Evaluasi

### Mean Average Precision (mAP)

mAP adalah metrik standar untuk evaluasi object detection. Ia menghitung rata-rata dari Average Precision (AP) di semua kelas.

**Precision-Recall Curve:**
- Precision: dari semua deteksi, berapa yang benar?
- Recall: dari semua objek, berapa yang berhasil dideteksi?

**AP** adalah area di bawah kurva precision-recall untuk satu kelas.

**mAP** = rata-rata AP di semua kelas.

Sering dilaporkan sebagai **mAP@0.5** (IoU threshold 0.5) atau **mAP@[0.5:0.95]** (rata-rata di berbagai IoU threshold, standar COCO).

---

## Implementasi

### Inference dengan YOLOv8 (Ultralytics)

```python
# Install: pip install ultralytics
from ultralytics import YOLO
import cv2

# Load model pre-trained
model = YOLO('yolov8n.pt')   # 'n' = nano (paling kecil, tercepat)
# Pilihan: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Inference pada gambar
results = model('path/to/image.jpg')

# Tampilkan hasil
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Koordinat bounding box
        confidence = box.conf[0].item()          # Confidence score
        class_id   = int(box.cls[0].item())      # ID kelas
        class_name = model.names[class_id]        # Nama kelas
        
        print(f"{class_name}: {confidence:.2f} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

# Simpan gambar dengan bounding box
result_image = results[0].plot()
cv2.imwrite('output.jpg', result_image)
```

### Fine-tuning YOLO pada Dataset Kustom

```python
from ultralytics import YOLO

# Load model base
model = YOLO('yolov8n.pt')

# Fine-tune pada dataset kustom
# Dataset harus dalam format YOLO: folder images/ dan labels/
results = model.train(
    data='path/to/dataset.yaml',  # File konfigurasi dataset
    epochs=100,
    imgsz=640,
    batch=16,
    name='my_custom_model'
)

# Evaluasi
metrics = model.val()
print(f"mAP@0.5:     {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")

# Export untuk deployment
model.export(format='onnx')   # Format ONNX untuk cross-platform
```

### Format File Konfigurasi Dataset (dataset.yaml)

```yaml
path: /path/to/dataset   # Root directory
train: images/train       # Training images
val:   images/val         # Validation images

nc: 3   # Jumlah kelas
names: ['kucing', 'anjing', 'burung']
```

### Format Label YOLO (setiap baris = satu objek)

```
# Format: class_id x_center y_center width height (semua dinormalisasi 0-1)
0 0.5 0.4 0.3 0.6    # kucing di tengah
1 0.2 0.3 0.15 0.3   # anjing di kiri
```

---

## Referensi

- [YOLO Paper — Redmon et al., 2016](https://arxiv.org/abs/1506.02640)
- [Faster R-CNN — Ren et al., 2015](https://arxiv.org/abs/1506.01497)
- [SSD — Liu et al., 2016](https://arxiv.org/abs/1512.02325)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [COCO Benchmark](https://cocodataset.org/#detection-eval)
- [A Survey on Object Detection — Zou et al.](https://arxiv.org/abs/1905.05055)
