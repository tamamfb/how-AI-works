# Machine Learning

> Modul ini membahas konsep dasar hingga lanjutan dalam Machine Learning — dari persiapan data, algoritma pembelajaran terawasi, tanpa pengawas, hingga pembelajaran berbasis penguatan.

---

## Struktur Modul

```
1. Machine Learning/
├── 1. Data/
├── 2. Supervised Learning/
├── 3. Unsupervised Learning/
└── 4. Reinforcement Learning/
```

---

## Daftar Materi

### [1. Data](./1.%20Data/)
> *"Garbage in, garbage out."*

Fondasi dari semua model ML adalah data yang berkualitas. Di sini kita akan belajar bagaimana mengumpulkan, membersihkan, dan mempersiapkan data sebelum dimasukkan ke model.

| Topik | Deskripsi |
|---|---|
| 📊 Exploratory Data Analysis (EDA) | Memahami distribusi dan pola dalam data |
| 🧹 Data Cleaning | Menangani missing values, outliers, dan duplikasi |
| 🔄 Data Preprocessing | Normalisasi, encoding, dan feature engineering |
| ✂️ Train/Test Split | Strategi pembagian dataset |

---

### [2. Supervised Learning](./2.%20Supervised%20Learning/)
> *"Belajar dari contoh yang sudah diberi label."*

Supervised Learning adalah paradigma dalam pembelajaran mesin yang menggunakan data berlabel untuk melatih algoritma matematis. Tujuannya adalah agar algoritma mempelajari hubungan antara input (fitur) dengan output (target) sehingga dapat secara akurat memprediksi output untuk data input yang belum terlihat.

#### Klasifikasi

Klasifikasi merupakan subset dari supervised learning, yang mana tugasnya adalah mengkategorikan data ke dalam kelas yang ditetapkan.

| Algoritma | Deskripsi |
|---|---|
| 👥 K-Nearest Neighbors | Klasifikasi berdasarkan kedekatan dengan tetangga terdekat |
| 🎲 Naive Bayes | Klasifikasi probabilistik berbasis Teorema Bayes |
| 📉 Logistic Regression | Klasifikasi biner dan multi-kelas |
| 🌳 Decision Tree | Klasifikasi berbasis pohon keputusan |
| 🌲 Random Forest | Ensemble dari banyak decision tree, lebih kuat dan tahan overfitting |
| 🔵 Support Vector Machine | Pemisahan kelas dengan hyperplane optimal |
| 🧠 Artificial Neural Network | Jaringan saraf tiruan terinspirasi dari otak manusia |

#### Regresi

Regresi merupakan subset dari supervised learning, yang mana tugasnya adalah memprediksi nilai kontinu berdasarkan input yang diberikan.

| Algoritma | Deskripsi |
|---|---|
| 📈 Linear Regression | Prediksi nilai kontinu dengan hubungan linear |
| 📐 Polynomial Regression | Regresi dengan hubungan non-linear berbentuk polinomial |
| 🎯 Ridge & Lasso Regression | Regresi dengan regularisasi untuk mencegah overfitting |
| 🌳 Decision Tree | Prediksi nilai kontinu berbasis pohon keputusan |
| 🔵 Support Vector Machine | Regresi dengan margin optimal (SVR) |

---

### [3. Unsupervised Learning](./3.%20Unsupervised%20Learning/)
> *"Menemukan pola tersembunyi tanpa label."*

Unsupervised Learning memungkinkan model untuk menemukan struktur dan pola dalam data tanpa petunjuk label dari manusia.

| Topik | Deskripsi |
|---|---|
| 🔵 K-Means Clustering | Pengelompokan data ke dalam K kluster |
| 🌿 Hierarchical Clustering | Klusterisasi berbasis hierarki |
| 🔶 DBSCAN | Klusterisasi berbasis densitas, tahan terhadap noise dan outlier |
| 🌸 BIRCH | Klusterisasi incremental untuk dataset besar |
| 📉 PCA (Principal Component Analysis) | Reduksi dimensi data |
| 🔍 Anomaly Detection | Mendeteksi data yang tidak normal |
| 🗺️ t-SNE & UMAP | Visualisasi data berdimensi tinggi |

---

### [4. Reinforcement Learning](./4.%20Reinforcement%20Learning/)
> *"Belajar dari interaksi dan konsekuensi tindakan."*

Reinforcement Learning adalah paradigma di mana agen belajar mengambil keputusan terbaik melalui interaksi dengan lingkungan, mendapat reward atau punishment.

| Topik | Deskripsi |
|---|---|
| 🏛️ Konsep Dasar RL | Agent, Environment, State, Action, Reward |
| 🎲 Q-Learning | Algoritma RL berbasis tabel nilai Q |
| 🧠 Deep Q-Network (DQN) | Q-Learning dengan Neural Network |
| 🎮 Policy Gradient | Optimasi kebijakan secara langsung |
| 🤝 Multi-Agent RL | Beberapa agen berinteraksi dalam lingkungan |

---

## 🛠️ Tools & Library yang Digunakan

| Library | Kegunaan |
|---|---|
| `NumPy` | Operasi numerik dan matriks |
| `Pandas` | Manipulasi dan analisis data |
| `Matplotlib` / `Seaborn` | Visualisasi data |
| `Scikit-learn` | Algoritma ML klasik |
| `TensorFlow` / `PyTorch` | Deep Learning & Neural Networks |
| `Gymnasium` | Lingkungan simulasi untuk Reinforcement Learning |

---

## Prasyarat

Sebelum memulai modul ini, pastikan kamu sudah memahami:
- Dasar-dasar **Python** (variabel, fungsi, loop)
- Konsep dasar **Matematika** (statistika, aljabar linear, kalkulus dasar)
- Penggunaan **Jupyter Notebook** atau IDE Python

---
