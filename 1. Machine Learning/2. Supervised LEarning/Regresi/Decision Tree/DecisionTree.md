# Decision Tree (Regresi)

## Daftar Isi

1. [Apa itu Decision Tree Regresi?](#apa-itu-decision-tree-regresi)
2. [Cara Kerja](#cara-kerja)
3. [Kriteria Pemisahan](#kriteria-pemisahan)
4. [Pruning](#pruning)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu Decision Tree Regresi?

**Decision Tree Regresi** menggunakan struktur pohon keputusan yang sama seperti klasifikasi, namun pada node daun (leaf) alih-alih memprediksi kelas, model memprediksi **nilai rata-rata** dari semua data training yang jatuh pada leaf tersebut.

> **Analogi:** Kamu mau prediksi harga rumah. Pertama tanya "Luas > 100m²?" → kalau iya, tanya "Ada garasi?" → kalau iya, prediksi harganya adalah **rata-rata** harga semua rumah di training yang punya luas > 100m² dan punya garasi. 🏠

---

## Cara Kerja

1. **Pilih fitur dan threshold** pemisah terbaik di setiap node
2. **Bagi data** menjadi dua subset berdasarkan threshold tersebut
3. **Ulangi** secara rekursif untuk setiap subset sampai kondisi berhenti terpenuhi
4. **Prediksi** = rata-rata nilai target di leaf node

Prediksi di leaf node $R_m$:

$$\hat{y} = \frac{1}{|R_m|} \sum_{i \in R_m} y_i$$

---

## Kriteria Pemisahan

Untuk regresi, pemisahan terbaik dipilih berdasarkan minimasi **Mean Squared Error** setelah split:

$$MSE_{split} = \frac{1}{n_L} \sum_{i \in L} (y_i - \bar{y}_L)^2 + \frac{1}{n_R} \sum_{i \in R} (y_i - \bar{y}_R)^2$$

| Kriteria | Keterangan |
|----------|-----------|
| **squared_error** | Minimasi MSE — default, paling umum |
| **absolute_error** | Minimasi MAE — lebih robust terhadap outlier |
| **friedman_mse** | Perbaikan MSE dengan koreksi Friedman |
| **poisson** | Untuk target count / rate (non-negatif) |

---

## Pruning

Tanpa batas, pohon akan tumbuh sangat dalam dan **overfit** — menghafal setiap data training.

**Hyperparameter pembatas:**

| Parameter | Fungsi | Nilai Umum |
|-----------|--------|-----------|
| **max_depth** | Kedalaman maksimum pohon | 3 sampai 10 |
| **min_samples_split** | Min sampel untuk memisah node | 10 sampai 50 |
| **min_samples_leaf** | Min sampel di setiap leaf | 5 sampai 20 |
| **max_leaf_nodes** | Batas maksimum jumlah leaf | 10 sampai 100 |

---

## Implementasi

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Inisialisasi dan training
dt = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt.fit(X_train, y_train)

# 2. Prediksi
y_pred = dt.predict(X_test)

# 3. Evaluasi
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"R²    : {r2:.4f}")
print(f"Depth : {dt.get_depth()}")

# 4. Visualisasi pohon
plt.figure(figsize=(20, 8))
plot_tree(
    dt,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title('Decision Tree Regressor')
plt.tight_layout()
plt.show()

# 5. Visualisasi prediksi vs aktual (untuk 1 fitur)
plt.figure(figsize=(8, 5))
X_line = np.linspace(X_test.min(), X_test.max(), 500).reshape(-1, 1)
y_line = dt.predict(X_line)
plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
plt.step(X_line, y_line, color='red', linewidth=2, label='Predicted', where='post')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Decision Tree Regression')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Feature importance
for name, imp in sorted(zip(feature_names, dt.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{name}: {imp:.4f}")
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Mudah divisualisasikan dan diinterpretasi | Prediksi berupa nilai rata-rata — tidak bisa ekstrapolasi |
| Bisa menangkap hubungan non-linear | Rentan overfit tanpa pruning |
| Tidak butuh feature scaling | Tidak stabil — perubahan kecil data bisa mengubah pohon drastis |
| Bisa handle fitur numerik dan kategorikal | Prediksi berbentuk "tangga" (step function) |

---

## Referensi

- [Scikit-learn: DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [StatQuest: Decision Trees for Regression](https://www.youtube.com/watch?v=g9c66TUylZ4)
