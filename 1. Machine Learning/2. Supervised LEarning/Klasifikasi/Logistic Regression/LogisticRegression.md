# Logistic Regression

## Daftar Isi

1. [Apa itu Logistic Regression?](#apa-itu-logistic-regression)
2. [Fungsi Sigmoid](#fungsi-sigmoid)
3. [Decision Boundary](#decision-boundary)
4. [Multi-kelas](#multi-kelas)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu Logistic Regression?

**Logistic Regression** adalah algoritma klasifikasi (bukan regresi!) yang memprediksi **probabilitas** suatu data masuk ke kelas tertentu. Outputnya selalu antara 0 dan 1, dihasilkan oleh **fungsi sigmoid**.

> **Analogi:** Bayangin kamu mau prediksi apakah email itu spam atau bukan. Logistic Regression tidak langsung jawab "iya/tidak", tapi bilang "ada 85% kemungkinan ini spam". Kalau lebih dari 50%, diklasifikasikan sebagai spam. 📧

---

## Fungsi Sigmoid

Kunci dari Logistic Regression adalah **fungsi sigmoid** yang mengubah nilai linear menjadi probabilitas:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Di mana $z = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n$ (kombinasi linear dari fitur-fitur).

| Nilai $z$ | Output Sigmoid | Interpretasi |
|-----------|----------------|--------------|
| $z \to +\infty$ | $\sigma(z) \to 1$ | Kelas positif pasti |
| $z = 0$ | $\sigma(z) = 0.5$ | Di batas keputusan |
| $z \to -\infty$ | $\sigma(z) \to 0$ | Kelas negatif pasti |

Model dilatih dengan meminimalkan **Binary Cross-Entropy Loss**:

$$L = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]$$

---

## Decision Boundary

Batas keputusan (decision boundary) adalah titik di mana $\sigma(z) = 0.5$, yang terjadi saat $z = 0$. Setiap data dengan $z > 0$ diprediksi kelas 1, sebaliknya kelas 0.

Decision boundary Logistic Regression bersifat **linear** di ruang fitur asli. Untuk batas non-linear, bisa tambahkan fitur polinomial.

---

## Multi-kelas

Untuk lebih dari 2 kelas, Logistic Regression menggunakan strategi:

| Strategi | Cara Kerja | Parameter |
|----------|-----------|-----------|
| **One-vs-Rest (OvR)** | Buat 1 model binary per kelas | `multi_class='ovr'` |
| **Softmax (Multinomial)** | 1 model untuk semua kelas sekaligus | `multi_class='multinomial'` |

Fungsi **Softmax** untuk multi-kelas:

$$P(C_k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

---

## Implementasi

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Scaling (penting untuk Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 2. Inisialisasi dan training
lr = LogisticRegression(
    C=1.0,              # Inverse regularization (C kecil = regularisasi kuat)
    penalty='l2',       # Jenis regularisasi: 'l1', 'l2', 'elasticnet'
    solver='lbfgs',     # Algoritma optimasi
    max_iter=1000,      # Iterasi maksimum
    multi_class='auto'  # 'auto', 'ovr', 'multinomial'
)
lr.fit(X_train_scaled, y_train)

# 3. Prediksi dan probabilitas
y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)

# 4. Evaluasi
print(f"Akurasi : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 5. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 6. Koefisien model
print("Koefisien:", lr.coef_)
print("Intercept:", lr.intercept_)
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Output berupa probabilitas, mudah diinterpretasi | Decision boundary linear — lemah untuk data non-linear |
| Training cepat | Rentan multikolinearitas antar fitur |
| Tidak sensitif terhadap outlier kecil | Butuh feature scaling |
| Regularisasi bawaan untuk cegah overfitting | Perlu banyak data untuk hasil yang stabil |

---

## Referensi

- [Scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
