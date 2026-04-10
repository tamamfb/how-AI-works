# Ridge & Lasso Regression

## Daftar Isi

1. [Apa itu Ridge & Lasso?](#apa-itu-ridge--lasso)
2. [Masalah Overfitting](#masalah-overfitting)
3. [Ridge Regression (L2)](#ridge-regression-l2)
4. [Lasso Regression (L1)](#lasso-regression-l1)
5. [Elastic Net](#elastic-net)
6. [Implementasi](#implementasi)
7. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu Ridge & Lasso?

**Ridge** dan **Lasso** adalah varian Linear Regression yang menambahkan **regularisasi** — penalti ekstra pada koefisien yang terlalu besar — untuk mencegah overfitting. Keduanya memaksa model agar lebih simpel.

> **Analogi:** Bayangin kamu menulis esai dan dibatasi panjangnya. Ridge seperti batasan yang mengecilkan ukuran huruf secara merata. Lasso seperti batasan yang memaksa kamu menghapus kalimat yang tidak penting sama sekali — lebih agresif, tapi hasilnya lebih bersih! 📝

---

## Masalah Overfitting

Pada Linear Regression biasa, model bisa punya koefisien yang sangat besar (terutama pada data berdimensi tinggi atau fitur berkorelasi tinggi). Koefisien besar → model terlalu spesifik ke training data → overfitting.

Regularisasi mengatasi ini dengan menambahkan **penalti** pada loss function:

$$L_{regularized} = L_{original} + \lambda \cdot \text{Penalty}$$

---

## Ridge Regression (L2)

**Ridge** menambahkan penalti **L2** (jumlah kuadrat koefisien):

$$L_{Ridge} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p w_j^2$$

**Efek:** Semua koefisien mengecil mendekati nol, tapi **tidak pernah persis nol**.

| $\lambda$ kecil | $\lambda$ besar |
|-----------------|-----------------|
| Mendekati Linear Regression biasa | Koefisien semakin kecil, model semakin sederhana |
| Mungkin overfit | Mungkin underfit |

**Cocok untuk:** Semua fitur relevan, ingin menstabilkan estimasi.

---

## Lasso Regression (L1)

**Lasso** menambahkan penalti **L1** (jumlah nilai absolut koefisien):

$$L_{Lasso} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |w_j|$$

**Efek:** Koefisien bisa menjadi **persis nol** → Lasso melakukan **feature selection otomatis**.

| | Ridge | Lasso |
|---|---|---|
| Penalti | $\sum w_j^2$ | $\sum \|w_j\|$ |
| Koefisien bisa nol? | ❌ Tidak | ✅ Ya |
| Feature selection | ❌ | ✅ Otomatis |
| Cocok untuk | Semua fitur relevan | Banyak fitur, ingin seleksi otomatis |

---

## Elastic Net

**Elastic Net** menggabungkan Ridge dan Lasso:

$$L_{ElasticNet} = \sum(y_i - \hat{y}_i)^2 + \lambda_1 \sum|w_j| + \lambda_2 \sum w_j^2$$

Parameter `l1_ratio` mengontrol proporsi L1 vs L2: `l1_ratio=0` → Ridge, `l1_ratio=1` → Lasso.

---

## Implementasi

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Scaling wajib karena regularisasi sensitif terhadap skala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---- Ridge ----
ridge = Ridge(alpha=1.0)  # alpha = lambda (parameter regularisasi)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# ---- Lasso ----
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# ---- Evaluasi ----
for name, y_pred in [('Ridge', y_pred_ridge), ('Lasso', y_pred_lasso)]:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"{name} -> RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ---- Mencari alpha optimal dengan Cross-Validation ----
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Best alpha Ridge: {ridge_cv.alpha_}")

lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Best alpha Lasso: {lasso_cv.alpha_:.4f}")

# ---- Visualisasi koefisien ----
plt.figure(figsize=(10, 4))
x_pos = range(len(feature_names))
plt.bar([p - 0.2 for p in x_pos], ridge.coef_, width=0.4, label='Ridge', alpha=0.7)
plt.bar([p + 0.2 for p in x_pos], lasso.coef_, width=0.4, label='Lasso', alpha=0.7)
plt.xticks(x_pos, feature_names, rotation=45)
plt.axhline(0, color='black', linewidth=0.5)
plt.title('Perbandingan Koefisien Ridge vs Lasso')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Kelebihan & Kekurangan

| | ✅ Kelebihan | ❌ Kekurangan |
|---|-------------|--------------|
| **Ridge** | Stabil saat fitur berkorelasi tinggi | Tidak melakukan feature selection |
| **Ridge** | Semua fitur tetap berkontribusi | Alpha harus di-tune |
| **Lasso** | Feature selection otomatis | Tidak stabil pada fitur berkorelasi tinggi |
| **Lasso** | Model lebih sparse dan interpretable | Hanya pilih satu fitur dari grup berkorelasi |
| **Elastic Net** | Gabungan keunggulan Ridge & Lasso | Dua hyperparameter untuk di-tune |

---

## Referensi

- [Scikit-learn: Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Scikit-learn: Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [StatQuest: Ridge Regression](https://www.youtube.com/watch?v=Q81RR3yKn30)
- [StatQuest: Lasso Regression](https://www.youtube.com/watch?v=NGf0voTMlcs)
