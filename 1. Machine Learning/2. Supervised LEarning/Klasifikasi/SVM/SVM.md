# Support Vector Machine (SVM)

## Daftar Isi

1. [Apa itu SVM?](#apa-itu-svm)
2. [Hyperplane dan Margin](#hyperplane-dan-margin)
3. [Kernel Trick](#kernel-trick)
4. [Parameter Penting](#parameter-penting)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu SVM?

**Support Vector Machine (SVM)** adalah algoritma klasifikasi yang mencari **hyperplane optimal** — garis (atau bidang) pemisah yang memaksimalkan jarak (margin) antara dua kelas. Data yang berada paling dekat dengan hyperplane disebut **support vector**.

> **Analogi:** Kamu punya dua kelompok titik di kertas (merah dan biru). Kamu harus gambar garis pemisah di antara keduanya. SVM tidak hanya mencari garis yang memisahkan, tapi garis yang punya **jarak maksimum** ke titik merah dan biru terdekat. Makin lebar jalannya, makin aman! 🛣️

---

## Hyperplane dan Margin

**Hyperplane** di ruang $n$ dimensi adalah: $w \cdot x + b = 0$

**Margin** adalah jarak total antara dua kelas:

$$\text{Margin} = \frac{2}{\|w\|}$$

SVM memaksimalkan margin dengan meminimalkan $\|w\|^2$, tunduk pada constraint:

$$y_i (w \cdot x_i + b) \geq 1 \quad \forall i$$

### Soft Margin (C-SVM)

Data nyata sering tidak linearly separable. Parameter **C** mengontrol trade-off antara margin lebar dan kesalahan klasifikasi:

| C kecil | C besar |
|---------|---------|
| Margin lebar, toleransi error lebih besar | Margin sempit, berusaha klasifikasi semua data benar |
| Lebih general (tahan overfitting) | Bisa overfit |

---

## Kernel Trick

Ketika data tidak bisa dipisahkan secara linear, SVM menggunakan **kernel** untuk memetakan data ke dimensi yang lebih tinggi di mana data bisa dipisahkan secara linear — tanpa menghitung transformasinya secara eksplisit!

| Kernel | Rumus | Kapan Dipakai |
|--------|-------|---------------|
| **Linear** | $K(x_i, x_j) = x_i \cdot x_j$ | Data linearly separable |
| **Polynomial** | $K(x_i, x_j) = (\gamma x_i \cdot x_j + r)^d$ | Data dengan hubungan polinomial |
| **RBF (Gaussian)** | $K(x_i, x_j) = e^{-\gamma \|x_i - x_j\|^2}$ | ⭐ Default, serbaguna |
| **Sigmoid** | $K(x_i, x_j) = \tanh(\gamma x_i \cdot x_j + r)$ | Mirip neural network |

---

## Parameter Penting

| Parameter | Fungsi | Default |
|-----------|--------|---------|
| **C** | Penalti kesalahan klasifikasi | 1.0 |
| **kernel** | Jenis kernel | 'rbf' |
| **gamma** | Pengaruh satu sampel training ('scale' atau 'auto' atau float) | 'scale' |
| **degree** | Derajat polinomial (hanya untuk kernel='poly') | 3 |

---

## Implementasi

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# 1. Scaling WAJIB untuk SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 2. Inisialisasi dan training
svm = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=True,  # Aktifkan jika butuh predict_proba
    random_state=42
)
svm.fit(X_train_scaled, y_train)

# 3. Prediksi
y_pred = svm.predict(X_test_scaled)

# 4. Evaluasi
print(f"Akurasi : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(f"Support Vectors: {svm.n_support_}")

# 5. Hyperparameter tuning dengan GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score : {grid.best_score_:.4f}")
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Efektif di ruang berdimensi tinggi | Lambat untuk dataset besar — O(n²) sampai O(n³) |
| Tahan overfitting karena memaksimalkan margin | Sangat sensitif terhadap skala fitur |
| Fleksibel dengan berbagai kernel | Sulit diinterpretasi (black box) |
| Efektif saat jumlah fitur > jumlah sampel | Pemilihan kernel dan C, gamma butuh tuning cermat |

---

## Referensi

- [Scikit-learn: SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [StatQuest: SVM](https://www.youtube.com/watch?v=efR1C6CvhmE)
