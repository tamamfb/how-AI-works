# Polynomial Regression

## Daftar Isi

1. [Apa itu Polynomial Regression?](#apa-itu-polynomial-regression)
2. [Persamaan Polinomial](#persamaan-polinomial)
3. [Degree dan Overfitting](#degree-dan-overfitting)
4. [Metrik Evaluasi](#metrik-evaluasi)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu Polynomial Regression?

**Polynomial Regression** adalah ekstensi dari Linear Regression yang mampu menangkap hubungan **non-linear** antara fitur dan target. Caranya: mengubah fitur asli menjadi fitur polinomial (pangkat 2, 3, dst.), lalu tetap menerapkan Linear Regression pada fitur yang sudah ditransformasi.

> **Analogi:** Jalanan tol yang lurus bisa diprediksi dengan garis lurus. Tapi jalanan berliku pegunungan butuh kurva. Polynomial Regression adalah cara menggambar kurva melalui titik-titik data yang tidak bisa dilalui garis lurus! 🏔️

---

## Persamaan Polinomial

Untuk satu fitur $x$ dengan degree $d$:

$$\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \ldots + w_d x^d$$

Transformasinya: satu fitur $x$ diubah menjadi $d$ fitur: $[x, x^2, x^3, \ldots, x^d]$

Untuk dua fitur $x_1, x_2$ dengan degree $2$, fitur yang dihasilkan:

$$[x_1,\ x_2,\ x_1^2,\ x_1 x_2,\ x_2^2]$$

> Jumlah fitur baru tumbuh cepat! Untuk $n$ fitur dan degree $d$, jumlah fitur menjadi $\binom{n+d}{d}$.

---

## Degree dan Overfitting

Pemilihan **degree** sangat kritis:

| Degree | Kondisi | Hasil |
|--------|---------|-------|
| Terlalu rendah | **Underfitting** | Model terlalu sederhana, tidak menangkap pola |
| Tepat | **Good fit** | Model menggeneralisasi dengan baik |
| Terlalu tinggi | **Overfitting** | Model hafal data training, buruk di data baru |

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

degrees = [1, 2, 5, 10]
plt.figure(figsize=(14, 4))

for i, d in enumerate(degrees):
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression().fit(X_poly, y_train)

    X_test_poly = poly.transform(X_test)
    r2 = r2_score(y_test, model.predict(X_test_poly))

    plt.subplot(1, 4, i + 1)
    plt.scatter(X_test, y_test, alpha=0.5, s=20)
    plt.title(f'Degree {d}\nR²={r2:.3f}')
plt.tight_layout()
plt.show()
```

---

## Metrik Evaluasi

| Metrik | Rumus | Keterangan |
|--------|-------|-----------|
| **MAE** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | Rata-rata error absolut |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Penalti lebih besar untuk error besar |
| **RMSE** | $\sqrt{MSE}$ | Sama satuan dengan target |
| **R² Score** | $1 - \frac{SS_{res}}{SS_{tot}}$ | 1 = sempurna, bisa negatif jika sangat buruk |

---

## Implementasi

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Buat pipeline Polynomial + Linear Regression
degree = 3
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('linear', LinearRegression())
])

# 2. Training
poly_model.fit(X_train, y_train)

# 3. Prediksi
y_pred = poly_model.predict(X_test)

# 4. Evaluasi
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# 5. Visualisasi kurva (untuk 1 fitur)
X_line = np.linspace(X_test.min(), X_test.max(), 300).reshape(-1, 1)
y_line = poly_model.predict(X_line)

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
plt.plot(X_line, y_line, color='red', linewidth=2, label=f'Degree {degree}')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title(f'Polynomial Regression (degree={degree})')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Bisa menangkap hubungan non-linear | Sangat mudah overfit pada degree tinggi |
| Tetap menggunakan Linear Regression (cepat) | Jumlah fitur tumbuh secara eksponensial |
| Mudah diimplementasikan dengan pipeline | Ekstrapolasi di luar range data training sangat tidak akurat |
| Degree bisa disesuaikan dengan kerumitan data | Sensitif terhadap outlier |

---

## Referensi

- [Scikit-learn: PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [StatQuest: Polynomial Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
