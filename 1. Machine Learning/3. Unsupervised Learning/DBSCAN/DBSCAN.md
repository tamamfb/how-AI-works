# DBSCAN

## Daftar Isi

- [Apa itu DBSCAN?](#apa-itu-dbscan)
- [Konsep Utama](#konsep-utama)
- [Cara Kerja](#cara-kerja)
- [Menentukan Parameter](#menentukan-parameter)
- [Implementasi](#implementasi)
- [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu DBSCAN?

**DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*) adalah algoritma clustering berbasis **kepadatan (density)**. Berbeda dari K-Means yang asumsi cluster-nya bulat, DBSCAN bisa menemukan cluster dengan **bentuk apapun** — dan yang paling keren, bisa otomatis mendeteksi **outlier/noise**.

> **Analogi:** Bayangin kamu lihat kerumunan orang di mall. Ada kerumunan besar di food court, kerumunan kecil di toko baju, dan satu-dua orang yang jalan sendirian. DBSCAN mengidentifikasi kerumunan padat sebagai cluster, dan orang-orang sendirian itu sebagai **noise**. 🏬

---

## Konsep Utama

DBSCAN punya dua parameter kunci:

| Parameter | Arti |
|-----------|------|
| **ε (eps)** | Radius lingkaran di sekitar suatu titik |
| **min_samples** | Jumlah minimum tetangga dalam radius ε agar dianggap "padat" |

Dari situ, setiap titik diklasifikasikan jadi 3 jenis:

| Jenis Titik | Definisi |
|-------------|----------|
| 🔵 **Core Point** | Punya ≥ `min_samples` tetangga dalam radius `eps` |
| 🟡 **Border Point** | Punya tetangga < `min_samples`, tapi masuk lingkaran core point |
| 🔴 **Noise Point** | Tidak termasuk core maupun border — ini **outlier**! |

---

## Cara Kerja

1. Pilih sembarang titik yang belum dikunjungi
2. Cari semua tetangga dalam radius `eps`
   - Kalau ≥ `min_samples` → tandai sebagai **Core Point**, buat cluster baru
   - Kalau < `min_samples` → tandai sementara sebagai **noise**
3. Ekspansi cluster dari core point — tambahkan semua tetangganya (termasuk core point lain dalam ε)
4. Ulangi sampai semua titik sudah dikunjungi

> Titik noise bisa "diselamatkan" kalau ternyata masuk dalam radius core point lain.

---

## Menentukan Parameter

### Menentukan `eps` — k-distance graph

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Cari jarak ke k tetangga terdekat (k = min_samples)
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)

# Sort jarak ke tetangga ke-k
k_distances = np.sort(distances[:, k-1])[::-1]

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.xlabel('Data Points (diurutkan)')
plt.ylabel(f'Jarak ke tetangga ke-{k}')
plt.title('k-Distance Graph')
plt.tight_layout()
plt.show()
# Titik "siku" pada grafik = nilai eps yang bagus!
```

> 💡 **Aturan umum:** `min_samples` ≥ dimensi data + 1. Untuk data 2D → min_samples ≥ 3, lebih aman pakai 4-5.

---

## Implementasi

```python
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Scaling (penting! DBSCAN berbasis jarak)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_scaled)

# 3. Info cluster
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)
print(f"Jumlah cluster : {n_clusters}")
print(f"Jumlah noise   : {n_noise}")

# 4. Evaluasi (jika ada lebih dari 1 cluster)
if n_clusters > 1:
    mask = labels != -1
    sil = silhouette_score(X_scaled[mask], labels[mask])
    print(f"Silhouette Score : {sil:.4f}")

# 5. Visualisasi
plt.figure(figsize=(8, 5))
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
for lbl, col in zip(unique_labels, colors):
    mask = labels == lbl
    marker = 'x' if lbl == -1 else 'o'
    label  = 'Noise' if lbl == -1 else f'Cluster {lbl}'
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                c=[col], marker=marker, s=50, label=label)
plt.title('DBSCAN Clustering')
plt.legend()
plt.tight_layout()
plt.show()
```

> ⚠️ Label `-1` dalam hasil `fit_predict` berarti titik tersebut dianggap **noise/outlier**.

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Tidak perlu tentukan jumlah cluster | Sensitif terhadap pemilihan `eps` dan `min_samples` |
| Bisa menemukan cluster bentuk apapun | Kesulitan menangani cluster dengan kepadatan berbeda |
| Otomatis mendeteksi outlier | Kurang efektif di data berdimensi tinggi |
| Tidak sensitif terhadap inisialisasi | Lambat untuk dataset sangat besar |

---

## Referensi

- [Scikit-learn: DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [StatQuest: DBSCAN](https://www.youtube.com/watch?v=RDZUdRSDOok)
