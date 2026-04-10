# K-Nearest Neighbors (KNN)

## Daftar Isi

1. [Apa itu KNN?](#apa-itu-knn)
2. [Cara Kerja](#cara-kerja)
3. [Menentukan Nilai K](#menentukan-nilai-k)
4. [Jenis Jarak](#jenis-jarak)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu KNN?

**K-Nearest Neighbors (KNN)** adalah algoritma klasifikasi yang paling intuitif — ketika ada data baru, algoritma ini mencari **K tetangga terdekat** di data training, lalu menentukan kelas berdasarkan **mayoritas kelas** dari tetangga tersebut.

> **Analogi:** Bayangin kamu pindah ke kota baru dan ingin tahu apakah suatu restoran itu bagus. Kamu tanya 5 tetangga terdekatmu — kalau 4 dari 5 bilang "enak", kamu simpulkan restoran itu bagus. Itulah KNN! 🏘️

---

## Cara Kerja

1. **Pilih nilai K** (jumlah tetangga yang akan dicek)
2. **Hitung jarak** antara data baru dan semua data training
3. **Ambil K tetangga** dengan jarak terkecil
4. **Vote mayoritas** — kelas yang paling banyak muncul di antara K tetangga = hasil prediksi

> Semakin kecil K, model semakin sensitif (bisa overfit). Semakin besar K, model semakin "smooth" tapi bisa underfit.

---

## Menentukan Nilai K

Nilai K optimal biasanya dicari dengan **cross-validation**. Aturan umum: coba nilai ganjil untuk menghindari seri suara, dan coba sekitar $\sqrt{n}$ sebagai titik awal.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_scores = []
k_range = range(1, 31)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

plt.figure(figsize=(8, 4))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi (CV)')
plt.title('Pemilihan Nilai K Optimal')
plt.tight_layout()
plt.show()
```

---

## Jenis Jarak

KNN mengukur "kedekatan" menggunakan rumus jarak. Yang paling umum:

| Jarak | Rumus | Kapan Dipakai |
|-------|-------|---------------|
| **Euclidean** | $\sqrt{\sum (x_i - y_i)^2}$ | Data kontinu, default |
| **Manhattan** | $\sum \|x_i - y_i\|$ | Data dengan banyak dimensi |
| **Minkowski** | $\left(\sum \|x_i - y_i\|^p\right)^{1/p}$ | Generalisasi Euclidean & Manhattan |

> 💡 **Penting:** Karena KNN berbasis jarak, **feature scaling wajib dilakukan** sebelum training. Gunakan `StandardScaler` atau `MinMaxScaler`.

---

## Implementasi

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 2. Inisialisasi dan training
knn = KNeighborsClassifier(
    n_neighbors=5,      # Jumlah tetangga
    metric='euclidean', # Jenis jarak
    weights='uniform'   # 'uniform' = semua tetangga sama bobot, 'distance' = bobot inversnya
)
knn.fit(X_train_scaled, y_train)

# 3. Prediksi
y_pred = knn.predict(X_test_scaled)

# 4. Evaluasi
print(f"Akurasi : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Simpel dan mudah dipahami | Lambat saat prediksi untuk data besar — hitung jarak ke semua data |
| Tidak ada proses training | Sangat sensitif terhadap skala fitur |
| Bisa dipakai untuk klasifikasi & regresi | Perlu memori besar untuk menyimpan seluruh data training |
| Tidak ada asumsi distribusi data | Buruk di data berdimensi tinggi (curse of dimensionality) |

---

## Referensi

- [Scikit-learn: KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [StatQuest: K-nearest neighbors](https://www.youtube.com/watch?v=HVXime0nQeI)
