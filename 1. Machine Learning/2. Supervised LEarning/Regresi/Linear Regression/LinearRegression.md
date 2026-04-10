# Linear Regression

## Daftar Isi

1. [Apa itu Linear Regression?](#apa-itu-linear-regression)
2. [Persamaan Garis](#persamaan-garis)
3. [Ordinary Least Squares](#ordinary-least-squares)
4. [Metrik Evaluasi](#metrik-evaluasi)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu Linear Regression?

**Linear Regression** adalah algoritma regresi paling fundamental yang memodelkan hubungan **linear** antara satu atau beberapa fitur input dengan nilai output kontinu. Tujuannya adalah mencari garis (atau hyperplane) yang paling "pas" dengan data.

> **Analogi:** Bayangin kamu mau prediksi harga rumah berdasarkan luasnya. Kamu plot semua data di grafik, lalu tarik garis lurus yang paling mendekati semua titik tersebut. Garis itulah model Linear Regression! 🏠📈

---

## Persamaan Garis

### Simple Linear Regression (1 fitur)

$$\hat{y} = w_0 + w_1 x$$

### Multiple Linear Regression (banyak fitur)

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n = \mathbf{w}^T \mathbf{x}$$

Di mana:
- $w_0$ = intercept (bias)
- $w_1, w_2, \ldots, w_n$ = koefisien / bobot setiap fitur
- $\hat{y}$ = nilai prediksi

---

## Ordinary Least Squares

Model dilatih dengan meminimalkan **Sum of Squared Errors (SSE)**:

$$L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

Solusi analitik (Ordinary Least Squares / OLS):

$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

> Solusi OLS langsung menghitung bobot optimal tanpa iterasi. Efisien untuk dataset kecil-menengah. Untuk dataset besar, gunakan **Stochastic Gradient Descent (SGDRegressor)**.

---

## Metrik Evaluasi

| Metrik | Rumus | Keterangan |
|--------|-------|-----------|
| **MAE** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | Rata-rata error absolut, robust terhadap outlier |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Penalti error besar lebih berat |
| **RMSE** | $\sqrt{MSE}$ | Sama satuan dengan target |
| **R² Score** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 1 = sempurna, 0 = tidak lebih baik dari rata-rata |

---

## Implementasi

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Inisialisasi dan training
lr = LinearRegression(
    fit_intercept=True  # Apakah model harus hitung intercept
)
lr.fit(X_train, y_train)

# 2. Prediksi
y_pred = lr.predict(X_test)

# 3. Evaluasi
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# 4. Koefisien
print(f"\nIntercept : {lr.intercept_:.4f}")
for name, coef in zip(feature_names, lr.coef_):
    print(f"{name:20s}: {coef:.4f}")

# 5. Visualisasi (untuk 1 fitur)
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Sangat simpel dan mudah diinterpretasi | Hanya bisa menangkap hubungan linear |
| Training sangat cepat | Sensitif terhadap outlier |
| Koefisien bisa menjelaskan pengaruh setiap fitur | Asumsi independensi, homoskedastisitas, normalitas residual |
| Tidak perlu feature scaling untuk OLS | Tidak cocok untuk hubungan non-linear tanpa feature engineering |

---

## Referensi

- [Scikit-learn: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=7ArmBVF2dCs)
