# BIRCH

## Daftar Isi

- [Apa itu BIRCH?](#apa-itu-birch)
- [Konsep Utama: CF Tree](#konsep-utama-cf-tree)
- [Parameter Penting](#parameter-penting)
- [Implementasi](#implementasi)
- [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu BIRCH?

**BIRCH** (*Balanced Iterative Reducing and Clustering using Hierarchies*) adalah algoritma clustering yang dirancang khusus untuk **dataset besar**. Daripada langsung memproses semua data, BIRCH merangkum data ke dalam struktur pohon kompak bernama **CF Tree**, sehingga lebih hemat memori dan lebih cepat.

> **Analogi:** Bayangin kamu harus menghitung rata-rata nilai 1 juta siswa, tapi komputermu tidak kuat. Daripada simpan semua nilai, kamu cukup catat: *"Ada 500 siswa dengan total nilai 35.000"*. Informasi ringkas ini cukup untuk hitung rata-rata tanpa harus ingat semuanya. Itulah ide di balik BIRCH! 📊

---

## Konsep Utama: CF Tree

BIRCH menggunakan **CF (Clustering Feature)** untuk merangkum sekumpulan data:

$$CF = (N, \overrightarrow{LS}, SS)$$

| Simbol | Arti |
|--------|------|
| **N** | Jumlah data dalam sub-cluster |
| **LS** | Linear Sum — jumlah semua vektor data |
| **SS** | Squared Sum — jumlah kuadrat semua vektor data |

Dari ketiga nilai ini, kita bisa menghitung **centroid** dan **radius** sub-cluster tanpa menyimpan data aslinya. Inilah kenapa BIRCH hemat memori!

**CF Tree** adalah pohon hierarki yang dibangun dari CF-CF tersebut. Data di-scan sekali, dan setiap titik baru dimasukkan ke sub-cluster yang paling dekat.

---

## Parameter Penting

| Parameter | Arti | Default |
|-----------|------|---------|
| **threshold** | Radius maksimum sub-cluster. Semakin kecil → lebih banyak sub-cluster, lebih akurat tapi lebih lambat | 0.5 |
| **branching_factor** | Jumlah maksimum anak per node di CF Tree | 50 |
| **n_clusters** | Jumlah cluster akhir (opsional, `None` berarti pakai hasil CF Tree langsung) | 3 |

> 💡 **Tips:** Mulai dengan `threshold=0.5`, lalu turunkan jika cluster terlalu kasar atau naikkan jika terlalu lambat.

---

## Implementasi

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit BIRCH
birch = Birch(threshold=0.5, branching_factor=50, n_clusters=3)
labels = birch.fit_predict(X_scaled)

# 3. Evaluasi
sil = silhouette_score(X_scaled, labels)
print(f"Silhouette Score : {sil:.4f}")
print(f"Jumlah cluster   : {len(set(labels))}")

# 4. Visualisasi (2D)
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', s=40)
plt.title('BIRCH Clustering')
plt.tight_layout()
plt.show()
```

### Partial Fit (Online Learning)

Keunggulan BIRCH adalah bisa dipakai untuk **data streaming** (datanya tidak harus tersedia sekaligus):

```python
birch = Birch(threshold=0.3, n_clusters=3)

# Proses data batch demi batch
for batch in np.array_split(X_scaled, 10):
    birch.partial_fit(batch)

labels = birch.predict(X_scaled)
```

---

## Perbandingan dengan K-Means

| | BIRCH | K-Means |
|---|---|---|
| **Kecepatan** | ⚡ Lebih cepat untuk data besar | Cukup cepat |
| **Memori** | Hemat (pakai CF Tree) | Lebih besar |
| **Data Streaming** | ✅ Mendukung `partial_fit` | ❌ Tidak |
| **Bentuk Cluster** | Cenderung spherical | Spherical |
| **Outlier** | Tidak otomatis terdeteksi | Tidak otomatis terdeteksi |
| **Cocok untuk** | Data besar (jutaan rows) | Data skala kecil-menengah |

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Sangat efisien untuk dataset besar | Cluster cenderung berbentuk spherical |
| Hemat memori — hanya simpan CF | Sensitif terhadap urutan data masuk |
| Mendukung online/streaming learning | Akurasi bisa di bawah K-Means untuk data kecil |
| Hanya perlu 1x scan data | Parameter `threshold` cukup tricky untuk di-tune |

---

## Referensi

- [Scikit-learn: Birch](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)
- [Paper Asli BIRCH (Zhang et al., 1996)](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf)
