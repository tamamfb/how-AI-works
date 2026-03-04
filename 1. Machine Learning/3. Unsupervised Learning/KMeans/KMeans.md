# K-Means Clustering

## Daftar Isi

- [Apa itu K-Means?](#apa-itu-k-means)
- [Cara Kerja](#cara-kerja)
- [Menentukan Nilai K](#menentukan-nilai-k-elbow-method)
- [Implementasi](#implementasi)
- [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu K-Means?

K-Means adalah algoritma clustering yang paling populer dan paling sering dipakai. Intinya, algoritma ini membagi data menjadi **K kelompok (cluster)**, di mana setiap data masuk ke cluster yang **paling dekat dengan pusatnya (centroid)**.

> **Analogi:** Bayangin kamu punya 30 orang di ruangan, terus diminta bagi jadi 3 grup berdasarkan posisi mereka. Kamu pilih 3 titik tengah secara acak, lalu setiap orang bergabung ke titik tengah yang paling dekat. Setelah itu titik tengahnya diupdate ke rata-rata posisi anggotanya — diulang terus sampai stabil. Itulah K-Means! 🧍🧍🧍

---

## Cara Kerja

K-Means bekerja secara **iteratif** melalui langkah-langkah berikut:

1. **Inisialisasi** → Pilih K centroid secara acak dari data
2. **Assignment** → Setiap data point ditetapkan ke centroid terdekat (pakai jarak Euclidean)
3. **Update** → Centroid diupdate menjadi rata-rata (mean) dari semua anggota cluster
4. **Ulangi** langkah 2-3 sampai centroid tidak berubah (konvergen)

$$d(x, c) = \sqrt{\sum_{i=1}^{n}(x_i - c_i)^2}$$

> Semakin kecil total jarak data ke centroid-nya (disebut **inertia** atau WCSS), semakin bagus clusteringnya.

---

## Menentukan Nilai K (Elbow Method)

Salah satu tantangan K-Means adalah: **berapa nilai K yang tepat?** 🤔

Jawabnya pakai **Elbow Method** — plot nilai inertia untuk berbagai K, lalu lihat di mana grafiknya "menyiku" (elbow). Titik siku itulah K optimal.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.tight_layout()
plt.show()
```

---

## Implementasi

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Scaling dulu (K-Means sensitif terhadap skala!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit K-Means
km = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels = km.fit_predict(X_scaled)

# 3. Evaluasi
sil = silhouette_score(X_scaled, labels)
print(f"Silhouette Score : {sil:.4f}")
print(f"Inertia          : {km.inertia_:.2f}")

# 4. Visualisasi (untuk 2D)
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', s=40)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            c='black', marker='X', s=200, label='Centroid')
plt.title('K-Means Clustering')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Simpel dan mudah dipahami | Harus tentukan K di awal |
| Cepat dan scalable untuk data besar | Sensitif terhadap outlier |
| Konvergensi terjamin | Hasil bisa berbeda tiap run (inisialisasi acak) |
| Interpretasi cluster mudah | Asumsi cluster berbentuk bulat (spherical) |

> 💡 **Tips:** Gunakan `KMeans(init='k-means++')` untuk inisialisasi centroid yang lebih cerdas dan hasil yang lebih konsisten!

---

## Referensi

- [Scikit-learn: KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [StatQuest: K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
