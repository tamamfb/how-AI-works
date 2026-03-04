# K-Means Clustering

## Daftar Isi

- [Apa itu K-Means?](#apa-itu-k-means)
- [Cara Kerja](#cara-kerja)
- [Menentukan Nilai K](#menentukan-nilai-k-elbow-method)
- [WCSS](#wcss-within-cluster-sum-of-squares)
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
2. **Assignment** → Setiap data point ditetapkan ke centroid terdekat (pakai jarak Euclidean / Manhattan)
3. **Update** → Centroid diupdate menjadi rata-rata (mean) dari semua anggota cluster
4. **Ulangi** langkah 2-3 sampai centroid tidak berubah (konvergen)

> Semakin kecil total jarak data ke centroid-nya (disebut **inertia** atau WCSS), semakin bagus clusteringnya.

---

## Menentukan Nilai K (Elbow Method)

Salah satu tantangan K-Means adalah: **berapa nilai K yang tepat?** 🤔

<img width="512" height="275" alt="image" src="https://github.com/user-attachments/assets/613c4694-4ee4-4cf5-b8a7-b3a16e013962" />

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

## WCSS (Within-Cluster Sum of Squares)

**WCSS** (atau *Inertia*) adalah metrik yang mengukur seberapa kompak cluster yang terbentuk. Rumusnya adalah jumlah kuadrat jarak setiap data ke centroid cluster-nya masing-masing:

$$WCSS = \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2$$

Di mana:
- $K$ = jumlah cluster
- $C_k$ = himpunan data dalam cluster ke-$k$
- $\mu_k$ = centroid cluster ke-$k$
- $\|x - \mu_k\|^2$ = kuadrat jarak Euclidean dari data $x$ ke centroid-nya

> **Intinya:** Semakin kecil WCSS, semakin data dalam satu cluster *berkumpul rapat* di sekitar centroidnya → clustering makin bagus.

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

---

## Referensi

- [Scikit-learn: KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [StatQuest: K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
