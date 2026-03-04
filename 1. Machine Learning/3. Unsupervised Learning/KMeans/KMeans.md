# K-Means Clustering

## Daftar Isi

- [Apa itu K-Means?](#apa-itu-k-means)
- [Cara Kerja](#cara-kerja)
- [Menentukan Nilai K](#menentukan-nilai-k-elbow-method)
- [WCSS](#wcss-within-cluster-sum-of-squares)
- [Silhouette Score](#silhouette-score)
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

## Silhouette Score

**Silhouette Score** adalah metrik evaluasi clustering yang mengukur seberapa baik setiap data masuk ke cluster-nya sendiri dibandingkan cluster lain. Nilainya berkisar antara **-1 sampai 1**.

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}$$

Di mana:
- $a(i)$ = rata-rata jarak titik $i$ ke semua titik **dalam cluster yang sama** (cohesion)
- $b(i)$ = rata-rata jarak titik $i$ ke semua titik **di cluster terdekat lainnya** (separation)

### Interpretasi Nilai

| Nilai | Artinya |
|-------|---------|
| Mendekati **1** | Data sudah di cluster yang tepat 🎯 |
| Sekitar **0** | Data berada di perbatasan antara dua cluster |
| Mendekati **-1** | Data kemungkinan masuk ke cluster yang salah ❌ |

> **Intinya:** Kita mau nilai silhouette yang **setinggi mungkin**. Kalau rendah, coba ganti jumlah cluster K-nya.

```python
from sklearn.metrics import silhouette_score

# Hitung setelah fit
labels = kmeans.labels_
sil = silhouette_score(X_train, labels)
print(f"Silhouette Score : {sil:.4f}")
```

---

## Implementasi

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Data train
X_train = [
    [50, 1], 
    [60, 2], 
    [80, 3], 
    [100, 3], 
    [120, 3], 
    [150, 4]
]

# Data uji / test
X_test = [
    [130, 3],
    [160, 4]
]

# 1. Inisialisasi dan melatih model K-Means
kmeans = KMeans(
    n_clusters=2,      # Jumlah klaster yang ingin dibentuk
    init="k-means++",  # Metode inisialisasi centroid
    n_init=10,         # Berapa kali algoritma dicoba untuk hasil terbaik
    max_iter=300,      # Iterasi maksimum
)
kmeans.fit(X_train)

# 2. Evaluasi
labels_train = kmeans.labels_
sil = silhouette_score(X_train, labels_train)
print(f"Silhouette Score : {sil:.4f}")
print(f"WCSS (Inertia)   : {kmeans.inertia_:.2f}")
print(f"Centroid         :\n{kmeans.cluster_centers_}")

# 3. Prediksi data uji
y_pred = kmeans.predict(X_test)
print(f"\nHasil prediksi X_test : {y_pred}")
# Output: [0] atau [1] — menunjukkan data uji masuk ke cluster mana

# 4. Visualisasi
plt.figure(figsize=(7, 5))
plt.scatter(
    np.array(X_train)[:, 0], np.array(X_train)[:, 1],
    c=labels_train, cmap='tab10', s=100, label='Train'
)
plt.scatter(
    np.array(X_test)[:, 0], np.array(X_test)[:, 1],
    c=y_pred, cmap='tab10', s=200, marker='*', edgecolors='black', label='Test'
)
plt.xlabel('Fitur 1')
plt.ylabel('Fitur 2')
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
