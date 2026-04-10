# Decision Tree (Klasifikasi)

## Daftar Isi

1. [Apa itu Decision Tree?](#apa-itu-decision-tree)
2. [Struktur Pohon](#struktur-pohon)
3. [Kriteria Pemisahan](#kriteria-pemisahan)
4. [Pruning](#pruning)
5. [Implementasi](#implementasi)
6. [Kelebihan & Kekurangan](#kelebihan--kekurangan)

---

## Apa itu Decision Tree?

**Decision Tree** adalah algoritma yang membangun model berbentuk **pohon keputusan** — setiap node adalah pertanyaan tentang suatu fitur, dan setiap daun adalah jawaban kelas. Cara kerjanya mirip permainan "20 pertanyaan".

> **Analogi:** Bayangin kamu mau tebak hewan. Pertama tanya "Apakah berkaki empat?" → kalau iya, tanya "Apakah bisa menggonggong?" → kalau iya, jawabannya "Anjing!". Decision tree bekerja persis seperti itu — serangkaian pertanyaan bercabang sampai dapat jawaban. 🐕

---

## Struktur Pohon

| Komponen | Definisi |
|----------|----------|
| **Root Node** | Node paling atas — fitur yang paling membedakan kelas |
| **Internal Node** | Pertanyaan tentang nilai suatu fitur |
| **Branch** | Jalur dari jawaban sebuah pertanyaan |
| **Leaf Node** | Node terminal — berisi prediksi kelas akhir |
| **Depth** | Kedalaman pohon dari root ke leaf terdalam |

---

## Kriteria Pemisahan

Decision tree memilih fitur pemisah terbaik menggunakan salah satu kriteria berikut:

### Gini Impurity

$$Gini(t) = 1 - \sum_{k=1}^{K} p_k^2$$

Di mana $p_k$ adalah proporsi kelas $k$ di node $t$. Nilai Gini = 0 berarti node murni (hanya satu kelas).

### Information Gain (Entropy)

$$Entropy(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

$$IG = Entropy(parent) - \sum_{child} \frac{n_{child}}{n_{parent}} \cdot Entropy(child)$$

| Kriteria | Default | Karakteristik |
|----------|---------|---------------|
| **gini** | ✅ Scikit-learn | Lebih cepat secara komputasi |
| **entropy** | | Lebih sensitif terhadap distribusi kelas |

---

## Pruning

Pohon yang terlalu dalam akan **overfit**. Ada dua cara membatasi kompleksitas:

**Pre-pruning** (sebelum tumbuh):

```python
DecisionTreeClassifier(
    max_depth=5,          # Kedalaman maksimum
    min_samples_split=10, # Min sampel untuk memisah node
    min_samples_leaf=5,   # Min sampel di setiap leaf
    max_features='sqrt'   # Jumlah fitur yang dipertimbangkan
)
```

**Post-pruning** (setelah tumbuh):

```python
# Cost-complexity pruning
path = clf.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas  # Cari alpha optimal via cross-validation
```

---

## Implementasi

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Inisialisasi dan training
dt = DecisionTreeClassifier(
    criterion='gini',     # 'gini' atau 'entropy'
    max_depth=5,          # Batasi kedalaman agar tidak overfit
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt.fit(X_train, y_train)

# 2. Prediksi
y_pred = dt.predict(X_test)

# 3. Evaluasi
print(f"Akurasi : {accuracy_score(y_test, y_pred):.4f}")
print(f"Depth   : {dt.get_depth()}")
print(classification_report(y_test, y_pred))

# 4. Visualisasi pohon
plt.figure(figsize=(20, 10))
plot_tree(
    dt,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree')
plt.tight_layout()
plt.show()

# 5. Feature importance
importances = dt.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {imp:.4f}")
```

---

## Kelebihan & Kekurangan

| ✅ Kelebihan | ❌ Kekurangan |
|-------------|--------------|
| Mudah divisualisasikan dan diinterpretasi | Rentan overfit jika tidak dipruning |
| Tidak butuh feature scaling | Tidak stabil — perubahan kecil data bisa mengubah pohon drastis |
| Bisa handle fitur numerik dan kategorikal | Bias terhadap fitur dengan banyak nilai unik |
| Tidak perlu asumsi distribusi data | Kurang akurat dibanding ensemble methods |

---

## Referensi

- [Scikit-learn: DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y)
