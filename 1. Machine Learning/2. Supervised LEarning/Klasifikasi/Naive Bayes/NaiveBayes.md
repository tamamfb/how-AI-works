# Naive Bayes

## Daftar Isi

1. [Apa itu Naive Bayes?](#apa-itu-naive-bayes)
2. [Teorema Bayes](#teorema-bayes)
3. [Asumsi Naive](#asumsi-naive)
4. [Jenis Naive Bayes](#jenis-naive-bayes)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu Naive Bayes?

**Naive Bayes** adalah algoritma klasifikasi berbasis probabilitas yang menerapkan **Teorema Bayes** dengan asumsi bahwa setiap fitur **saling independen** satu sama lain (inilah yang disebut "naive" / polos).

> **Analogi:** Bayangin kamu dokter yang mendiagnosis pasien. Kamu lihat gejala: demam, batuk, pilek. Kamu anggap setiap gejala independen dan hitung probabilitas penyakit berdasarkan kombinasi gejalanya. Dokter Bayes! 🩺

---

## Teorema Bayes

Inti dari algoritma ini adalah Teorema Bayes:

$$P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}$$

| Notasi | Arti |
|--------|------|
| $P(C \mid X)$ | Probabilitas kelas $C$ **setelah** melihat fitur $X$ (posterior) |
| $P(X \mid C)$ | Probabilitas fitur $X$ **jika** kelasnya $C$ (likelihood) |
| $P(C)$ | Probabilitas kelas $C$ secara umum (prior) |
| $P(X)$ | Probabilitas fitur $X$ (konstan, bisa diabaikan dalam perbandingan) |

Kelas yang dipilih adalah yang memiliki nilai **posterior tertinggi**:

$$\hat{C} = \arg\max_C \; P(C) \prod_{i=1}^n P(x_i \mid C)$$

---

## Asumsi Naive

Kata "naive" berasal dari asumsi bahwa **semua fitur independen** satu sama lain, padahal dalam kenyataan hal ini jarang benar. Meskipun demikian, Naive Bayes tetap bekerja sangat baik dalam praktik, terutama untuk:

- Klasifikasi teks (spam filter, analisis sentimen)
- Klasifikasi dokumen
- Data dengan banyak fitur

---

## Jenis Naive Bayes

| Jenis | Asumsi Distribusi | Cocok untuk |
|-------|-------------------|-------------|
| **GaussianNB** | Fitur kontinu berdistribusi normal (Gaussian) | Data numerik |
| **MultinomialNB** | Fitur berupa hitungan (count) | Klasifikasi teks (frekuensi kata) |
| **BernoulliNB** | Fitur berupa biner 0/1 | Klasifikasi teks (ada/tidaknya kata) |
| **ComplementNB** | Modifikasi MultinomialNB | Dataset tidak seimbang |

---

## Implementasi

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Inisialisasi dan training
gnb = GaussianNB(
    var_smoothing=1e-9  # Smoothing untuk mencegah probabilitas nol
)
gnb.fit(X_train, y_train)

# 2. Prediksi
y_pred = gnb.predict(X_test)

# 3. Probabilitas prediksi
y_prob = gnb.predict_proba(X_test)
print("Probabilitas kelas:", y_prob[:5])

# 4. Evaluasi
print(f"Akurasi : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Untuk Klasifikasi Teks

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Ubah teks ke vektor hitungan kata
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec  = vectorizer.transform(X_test_text)

mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
mnb.fit(X_train_vec, y_train)
y_pred = mnb.predict(X_test_vec)
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Sangat cepat dan efisien | Asumsi independensi fitur sering tidak realistis |
| Bekerja baik dengan sedikit data training | Tidak bisa menangkap hubungan antar fitur |
| Sangat baik untuk klasifikasi teks | Jika suatu nilai tidak ada di training, probabilitas jadi 0 (perlu smoothing) |
| Tidak sensitif terhadap fitur yang tidak relevan | Kurang akurat untuk fitur kontinu yang tidak normal |

---

## Referensi

- [Scikit-learn: GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
