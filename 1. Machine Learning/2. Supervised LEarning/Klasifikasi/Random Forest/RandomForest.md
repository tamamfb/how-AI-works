# Random Forest

## Daftar Isi

1. [Apa itu Random Forest?](#apa-itu-random-forest)
2. [Cara Kerja](#cara-kerja)
3. [Bootstrap & Bagging](#bootstrap--bagging)
4. [Feature Importance](#feature-importance)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu Random Forest?

**Random Forest** adalah algoritma **ensemble** yang membangun banyak **Decision Tree secara acak**, lalu menggabungkan hasilnya melalui voting mayoritas (klasifikasi) atau rata-rata (regresi). "Random" karena setiap pohon dilatih pada **sampel acak** data dengan **subset acak** fitur.

> **Analogi:** Daripada tanya satu dokter ahli, kamu tanya 100 dokter berbeda dan ambil pendapat mayoritas. Masing-masing dokter punya pengalaman sedikit berbeda — hasilnya jauh lebih reliable! Itulah Random Forest. 👨‍⚕️👩‍⚕️👨‍⚕️

---

## Cara Kerja

1. **Bootstrap Sampling** — untuk setiap pohon, ambil sampel acak **dengan pengembalian** dari data training (~63% data unik masuk, sisanya jadi Out-of-Bag)
2. **Random Feature Selection** — di setiap node pemisahan, hanya pertimbangkan **subset acak** fitur (bukan semua fitur)
3. **Grow Tree** — tumbuhkan setiap pohon hingga kedalaman penuh (atau sampai batas tertentu) tanpa pruning
4. **Agregasi** — hasil prediksi dikumpulkan dari semua pohon, diputuskan lewat **voting mayoritas**

---

## Bootstrap & Bagging

**Bootstrap Aggregating (Bagging)** adalah teknik yang membuat setiap pohon lebih beragam satu sama lain:

| Konsep | Penjelasan |
|--------|-----------|
| **Bootstrap** | Sampel dengan pengembalian — beberapa data muncul > 1 kali |
| **Out-of-Bag (OOB)** | ~37% data yang tidak masuk training setiap pohon — bisa dipakai untuk evaluasi tanpa butuh validation set |
| **Agregasi** | Gabungkan prediksi semua pohon → kurangi variance, tingkatkan akurasi |

```python
# OOB score bisa langsung dihitung
rf = RandomForestClassifier(oob_score=True, ...)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")
```

---

## Feature Importance

Salah satu keunggulan Random Forest adalah bisa mengukur **seberapa penting setiap fitur**:

$$Importance(f) = \frac{1}{T} \sum_{t=1}^T \sum_{\text{node splits on } f} \frac{n_{\text{node}}}{n_{\text{total}}} \cdot \Delta Gini$$

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(rf.feature_importances_, index=feature_names)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(8, 6))
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```

---

## Implementasi

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Inisialisasi dan training
rf = RandomForestClassifier(
    n_estimators=100,   # Jumlah pohon
    max_depth=None,     # None = tumbuh penuh, atau batasi dengan integer
    max_features='sqrt',# Jumlah fitur per split: 'sqrt', 'log2', atau integer
    min_samples_leaf=1,
    oob_score=True,     # Gunakan OOB untuk estimasi akurasi
    n_jobs=-1,          # Gunakan semua CPU core
    random_state=42
)
rf.fit(X_train, y_train)

# 2. Prediksi
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

# 3. Evaluasi
print(f"Akurasi  : {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")
print(classification_report(y_test, y_pred))
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Sangat akurat dan tahan overfitting | Lebih lambat dan butuh memori lebih dari Decision Tree tunggal |
| Bekerja baik tanpa banyak tuning | Kurang interpretable — sulit divisualisasikan |
| Bisa estimasi feature importance | Tidak efisien untuk data real-time / streaming |
| OOB score sebagai estimasi validasi gratis | Bisa overfit pada data dengan banyak noise |

---

## Referensi

- [Scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
