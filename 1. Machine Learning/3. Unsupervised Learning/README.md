# Sedikit pencerahan tentang Unsupervised Learning

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Pengenalan](#pengenalan)
- [Clustering](#clustering)
- [Referensi](#referensi)

## Pengenalan

Oke, sebelumnya kita udah kenalan sama **Supervised Learning** — model yang belajar dari data **berlabel**. Nah, *Unsupervised Learning* ini beda.

Unsupervised learning adalah paradigma dalam pembelajaran mesin yang menggunakan data **tidak berlabel** untuk melatih algoritma. Artinya, modelnya belajar sendiri tanpa ada "kunci jawaban" dari kita. Tujuannya bukan untuk memprediksi sesuatu yang spesifik, tapi untuk **menemukan pola atau struktur tersembunyi** dalam data.

---

### Analogi gampangnya:

Bayangin kamu dapat tugas: *"Kelompokkan 100 buah ini menjadi beberapa grup!"* — tapi nggak ada instruksi lebih lanjut. Kamu sendiri yang harus memutuskan: ini mirip-mirip, ini juga mirip, dst. Nah, itulah yang dilakukan Unsupervised Learning. 🍎🍊🍇

---

### Perbedaan dengan Supervised Learning

| | Supervised Learning | Unsupervised Learning |
|---|---|---|
| **Label** | Ada (berlabel) | Tidak ada (tidak berlabel) |
| **Tujuan** | Prediksi output | Menemukan pola/struktur |
| **Contoh tugas** | Klasifikasi, Regresi | Clustering, Dimensionality Reduction |
| **Evaluasi** | Mudah (bandingkan prediksi vs label) | Lebih sulit (tidak ada ground truth) |

---

## Clustering

Clustering merupakan subset dari unsupervised learning yang tugasnya adalah **mengelompokkan data berdasarkan kemiripannya**. Data yang mirip satu sama lain akan masuk ke satu kelompok (cluster), sedangkan data yang berbeda akan masuk ke kelompok lain.

> **Analogi:** Bayangin kamu punya banyak lagu di playlist, lalu kamu kelompokkan sendiri: ini rock, ini pop, ini jazz — padahal nggak ada yang ngasih label dulu. Algoritma clustering melakukan hal yang sama! 🎵

---

### Algoritma Clustering

Beberapa algoritma clustering yang umum dipakai:

| Algoritma | Kelebihan | Kekurangan |
|-----------|-----------|------------|
| **K-Means** | Cepat, simpel, scalable | Perlu tentukan K di awal, sensitif terhadap outlier |
| **Hierarchical** | Tidak perlu tentukan K di awal, menghasilkan dendrogram | Lambat untuk data besar |
| **DBSCAN** | Tahan terhadap outlier, bentuk cluster bebas | Sensitif terhadap parameter `eps` dan `min_samples` |
| **BIRCH** | Efisien untuk data sangat besar | Kurang akurat untuk cluster berbentuk non-spherical |

Klik untuk baca lebih lanjut:
- [K-Means](KMeans/KMeans.md)
- [Hierarchical](Hierarchical/Hierarchical.md)
- [DBSCAN](DBSCAN/DBSCAN.md)
- [BIRCH](BIRCH/BIRCH.md)

---

### Cara Evaluasi Clustering

Karena tidak ada label, kita pakai metrik khusus untuk evaluasi:

- **Silhouette Score** → Mengukur seberapa mirip suatu titik ke clusternya sendiri dibanding cluster lain. Nilainya antara -1 (buruk) sampai 1 (sempurna).
- **Davies-Bouldin Index** → Semakin kecil nilainya, semakin baik clustering-nya.
- **Inertia (WCSS)** → Dipakai di K-Means untuk metode Elbow.

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

sil = silhouette_score(X, labels)
db  = davies_bouldin_score(X, labels)

print(f"Silhouette Score : {sil:.4f}")
print(f"Davies-Bouldin   : {db:.4f}")
```

---

## Referensi

- [Scikit-learn: Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html)
- [Scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html)