# Hierarchical Clustering

## Daftar Isi

- [Apa itu Hierarchical Clustering?](#apa-itu-hierarchical-clustering)
- [Agglomerative vs Divisive](#agglomerative-vs-divisive)
- [Linkage Method](#linkage-method)
- [Dendrogram](#dendrogram)
- [Implementasi](#implementasi)
- [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu Hierarchical Clustering?

Hierarchical Clustering adalah algoritma clustering yang membangun **hierarki cluster** secara bertingkat — seperti pohon keluarga. Bedanya dari K-Means, kamu **tidak perlu tentukan jumlah cluster di awal**. Kamu cukup lihat hasil dendrogramnya, lalu potong di ketinggian yang kamu mau.

> **Analogi:** Bayangin kamu lagi bikin silsilah keluarga. Mulai dari individu, lalu satukan yang paling mirip jadi keluarga inti, lalu satukan keluarga inti jadi keluarga besar, dan seterusnya — sampai semuanya bersatu jadi satu pohon besar. 🌳

---

## Agglomerative vs Divisive

Ada dua pendekatan utama:

| | **Agglomerative** (Bottom-Up) | **Divisive** (Top-Down) |
|---|---|---|
| **Mulai dari** | Setiap data = 1 cluster | Semua data = 1 cluster |
| **Proses** | Gabungkan dua cluster terdekat | Pecah cluster menjadi sub-cluster |
| **Arah** | ⬆️ Bawah ke atas | ⬇️ Atas ke bawah |
| **Popularitas** | ⭐ Lebih umum dipakai | Jarang dipakai |

> Yang paling sering dipakai adalah **Agglomerative Clustering**.

---

## Linkage Method

*Linkage* menentukan bagaimana jarak antar **dua cluster** dihitung:

| Linkage | Cara Hitung Jarak | Cocok untuk |
|---------|-------------------|-------------|
| **Single** | Jarak dua titik terdekat | Cluster memanjang / chain-like |
| **Complete** | Jarak dua titik terjauh | Cluster kompak, ukuran seragam |
| **Average** | Rata-rata semua pasangan titik | Keseimbangan antara keduanya |
| **Ward** | Minimasi total varians dalam cluster | ⭐ Paling umum, cluster bulat kompak |

---

## Dendrogram

Dendrogram adalah visualisasi hierarki hasil clustering. Sumbu Y menunjukkan jarak (atau dissimilarity) antar cluster. Semakin tinggi garis penggabungan, semakin berbeda kedua cluster tersebut.

**Cara memotong dendrogram:**
- Tarik garis horizontal di ketinggian tertentu
- Jumlah cabang yang dipotong = jumlah cluster

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hitung linkage matrix
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=45)
plt.title('Dendrogram (Ward Linkage)')
plt.xlabel('Data Point / Cluster')
plt.ylabel('Jarak')
plt.tight_layout()
plt.show()
```

---

## Implementasi

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit model (tentukan n_clusters setelah lihat dendrogram)
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X_scaled)

# 3. Evaluasi
sil = silhouette_score(X_scaled, labels)
print(f"Silhouette Score : {sil:.4f}")

# 4. Visualisasi (2D)
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', s=40)
plt.title('Hierarchical Clustering')
plt.tight_layout()
plt.show()
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Tidak perlu tentukan K di awal | Lambat untuk dataset besar — O(n²) atau O(n³) |
| Dendrogram membantu visualisasi | Sekali cluster digabung, tidak bisa dipisah lagi |
| Tidak ada asumsi bentuk cluster | Sensitif terhadap outlier (terutama single linkage) |
| Cocok untuk data kecil-menengah | Memori besar untuk data berukuran besar |

---

## Referensi

- [Scikit-learn: AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo)
