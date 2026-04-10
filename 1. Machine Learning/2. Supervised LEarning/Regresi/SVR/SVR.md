# Support Vector Regression (SVR)

## Daftar Isi

1. [Apa itu SVR?](#apa-itu-svr)
2. [Epsilon-Insensitive Tube](#epsilon-insensitive-tube)
3. [Kernel pada SVR](#kernel-pada-svr)
4. [Parameter Penting](#parameter-penting)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu SVR?

**Support Vector Regression (SVR)** adalah adaptasi SVM untuk tugas regresi. Alih-alih mencari hyperplane pemisah kelas, SVR mencari **fungsi** yang memprediksi nilai kontinu sambil berusaha agar sebanyak mungkin data masuk dalam **tabung toleransi** lebar $2\varepsilon$ di sekitar garis prediksi.

> **Analogi:** Bayangin kamu menggambar garis prediksi harga saham. SVR tidak peduli error kecil (masih dalam tabung $\varepsilon$), tapi menghukum keras error yang melampaui batas tabung. Ini seperti tukang ukur yang toleran terhadap kesalahan kecil, tapi tidak untuk yang besar! 📊

---

## Epsilon-Insensitive Tube

SVR menggunakan **$\varepsilon$-insensitive loss function**:

$$L_\varepsilon(y, \hat{y}) = \begin{cases} 0 & \text{jika } |y - \hat{y}| \leq \varepsilon \\ |y - \hat{y}| - \varepsilon & \text{jika } |y - \hat{y}| > \varepsilon \end{cases}$$

**Tujuan optimasi SVR:**

$$\min \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$$

Di mana $\xi_i, \xi_i^*$ adalah **slack variables** untuk error yang melebihi $\varepsilon$, dan $C$ adalah parameter penalti.

| Di dalam tabung ($|\text{error}| \leq \varepsilon$) | Di luar tabung ($|\text{error}| > \varepsilon$) |
|-----------------------------------------------------|------------------------------------------------|
| Loss = 0, tidak ada penalti | Loss = $|\text{error}| - \varepsilon$ |
| Data ini tidak memengaruhi model | Data ini menjadi **support vector** |

---

## Kernel pada SVR

Sama seperti SVM klasifikasi, SVR mendukung berbagai kernel untuk hubungan non-linear:

| Kernel | Rumus | Kapan Dipakai |
|--------|-------|---------------|
| **Linear** | $K(x_i, x_j) = x_i \cdot x_j$ | Hubungan linear |
| **RBF** | $e^{-\gamma\|x_i-x_j\|^2}$ | ⭐ Default, fleksibel untuk non-linear |
| **Polynomial** | $(\gamma x_i \cdot x_j + r)^d$ | Hubungan polinomial |
| **Sigmoid** | $\tanh(\gamma x_i \cdot x_j + r)$ | Jarang dipakai |

---

## Parameter Penting

| Parameter | Fungsi | Nilai Umum |
|-----------|--------|-----------|
| **C** | Penalti untuk error di luar tabung. Besar = fit ketat | 1.0, 10, 100 |
| **epsilon** | Lebar setengah tabung toleransi | 0.1, 0.01 |
| **kernel** | Jenis kernel | 'rbf' |
| **gamma** | Pengaruh satu sampel (untuk RBF/poly) | 'scale', 'auto' |

**Hubungan C dan epsilon:**

| | C kecil | C besar |
|---|---------|---------|
| **epsilon kecil** | Sangat toleran, model simpel | Akurasi tinggi, bisa overfit |
| **epsilon besar** | Model sangat simpel, mungkin underfit | Lebih smooth |

---

## Implementasi

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# 1. Scaling WAJIB untuk SVR
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# 2. Inisialisasi dan training
svr = SVR(
    kernel='rbf',
    C=100,
    epsilon=0.1,
    gamma='scale'
)
svr.fit(X_train_scaled, y_train_scaled)

# 3. Prediksi (kembalikan ke skala asli)
y_pred_scaled = svr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# 4. Evaluasi
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"RMSE             : {rmse:.4f}")
print(f"MAE              : {mae:.4f}")
print(f"R²               : {r2:.4f}")
print(f"Support Vectors  : {svr.n_support_[0]}")

# 5. Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train_scaled, y_train_scaled)
print(f"Best params: {grid.best_params_}")

# 6. Visualisasi prediksi vs aktual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('SVR: Actual vs Predicted')
plt.tight_layout()
plt.show()
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Efektif di ruang berdimensi tinggi | Lambat untuk dataset besar |
| Robust terhadap outlier (berkat $\varepsilon$ tube) | Sangat sensitif terhadap skala fitur |
| Bisa menangkap hubungan non-linear via kernel | Banyak hyperparameter yang perlu di-tuning |
| Tidak terpengaruh data di dalam tabung | Sulit diinterpretasi |

---

## Referensi

- [Scikit-learn: SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [StatQuest: Support Vector Machines (SVR)](https://www.youtube.com/watch?v=efR1C6CvhmE)
