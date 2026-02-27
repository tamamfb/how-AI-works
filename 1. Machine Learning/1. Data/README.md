# Sedikit pencerahan tentang Data

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Pendahuluan](#pendahuluan)
- [Main Course](#main-course)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Train/Test Split](#4-traintest-split)

## Pendahuluan
Oke sebelum kita mulai, mari kita lihat bersama Alur sebuah model ML dibuat:

<img width="1050" height="500" alt="image" src="https://github.com/user-attachments/assets/90d17e98-6f8a-4535-b428-0de1bb340b20" />

Ya bisa dilihat digambar tersebut, bahwa salah satu (dua seh) step di ML Pipeline itu "berhubungan" dengan data. Tapi apakah data emang se urgent itu untuk kebagusan sebuah model?

---

### Mari kita lihat ilustrasi berikut:

<img width="1050" height="500" alt="image" src="https://github.com/user-attachments/assets/af0d3ee2-e37e-4857-8689-dc179a3f49c5" />

Anda diminta untuk predict angka selanjutnya. Oke kalau ini jawabannya udah pasti kan ya? **Yaitu 11**. Sebab polanya tinggal +2 saja :)

---

### Oke sekarang lihat ilustrasi 2 berikut:

<img width="1050" height="500" alt="image" src="https://github.com/user-attachments/assets/52a0a862-e8eb-49b8-a24d-2cb74e5ef52d" />

Nahh, apa angka selanjutnya? apakah 8?? apakah 7?? Kita ga punya cukup informasi untuk menentukan angka selanjutnya. Itulah mengapa Data yang banyak dapat meningkatkan kualitas sebuah model.

<img width="1050" height="500" alt="image" src="https://github.com/user-attachments/assets/5803a3bb-090b-46d1-9e66-f564e7877415" />

---

### Sekarang lihat ilustrasi 3 berikut:

<img width="1050" height="500" alt="image" src="https://github.com/user-attachments/assets/2dd3b6ae-e4c2-49c1-a800-5386ea6e27cd" />

Yap aku juga gatau jawabannya apa. bisa dilihat bahwa Data yang banyak tidak menjamin kualitas model yang bagus. Kebenaran data juga harus diperhatikan

---

### Kesimpulan:
Agar model ML kita bagus, kita butuh data yang banyak dan akurat. Semakin banyak dan semakin akurat data kita, maka semakin bagus juga model ML kita :)

---

## Main Course

Oke sekarang kita bahas hal-hal yang perlu dilakukan terhadap data sebelum dimasukkan ke model ML:

1. [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
2. [Data Cleaning](#2-data-cleaning)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Train/Test Split](#4-traintest-split)

Kita bakal pakai dataset **Titanic** dari Kaggle sebagai contoh!

---

## 1. Exploratory Data Analysis (EDA)

EDA adalah proses **kenalan dulu sama data** kita sebelum ngapa-ngapain. Tujuannya biar kita ngerti kondisi data: seberapa besar, tipe datanya apa, ada yang kosong gak, distribusinya kayak gimana, dll.

### Head & Shape

Lihat beberapa baris pertama dan ukuran dataset.

```python
df.head(5)
print("Shape:", df.shape)
```

### Tipe Data & Statistik Deskriptif

```python
df.dtypes
df.describe()
```

`describe()` langsung kasih ringkasan statistik (mean, std, min, max, dll.) untuk semua kolom numerik sekaligus.

### Distribusi Data

Cek distribusi setiap kolom numerik pakai **Histogram**, **KDE plot**, dan **Boxplot**. Dari sini kita bisa lihat apakah data normal, skewed, atau ada anomali.

### Missing Values

Cek apakah ada nilai kosong di dataset kita.

```python
missing = pd.DataFrame({
    'Count': df.isnull().sum(),
    '%'    : (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing[missing['Count'] > 0])
```

### Duplicate

Cek apakah ada baris yang sama persis.

```python
print("Jumlah duplikat:", df.duplicated().sum())
```

### Korelasi

Visualisasi hubungan antar kolom numerik pakai **heatmap**. Kalau dua fitur berkorelasi sangat tinggi, mungkin salah satunya bisa dihapus (redundant).

### Outlier

Deteksi outlier secara visual (boxplot) dan kuantitatif (IQR).

### Class Imbalance

Kalau ini supervised learning, cek apakah distribusi kelas target seimbang. Model bisa bias ke kelas mayoritas kalau datanya imbalanced!

---

## 2. Data Cleaning

Setelah tau kondisi data, sekarang waktunya **bersih-bersih**! 🧹

### Hapus Duplikat

```python
df = df.drop_duplicates()
```

### Tangani Missing Values

Ada beberapa strategi tergantung kondisi:

| Kondisi | Strategi |
|---------|----------|
| Missing > 50% di suatu kolom | Drop kolomnya |
| Missing di kolom target | Drop barisnya |
| Kolom numerik | Imputasi dengan **median** |
| Kolom kategori | Imputasi dengan **modus** |

```python
# Drop kolom jika missing > 50%
df = df.loc[:, df.isnull().mean() < 0.5]

# Imputasi numerik → median
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Imputasi kategori → modus
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])
```

### Hapus Outlier (IQR Method)

Outlier bisa ngerusak performa model. Kita pakai metode **IQR** untuk identifikasi dan hapus outlier:

$$\text{Outlier jika: } x < Q1 - 1.5 \times IQR \;\text{ atau }\; x > Q3 + 1.5 \times IQR$$

```python
for col in cols_for_outlier:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df  = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
```

### Hapus Kolom Tidak Relevan

Kolom seperti ID, nama, nomor tiket, dll. biasanya tidak berkontribusi pada prediksi.

```python
df = df.drop(columns=['PassengerId', 'Name'])
```

### Samakan Kategori

Pastikan nilai kategori konsisten. `'male'` dan `'Male'` harus dianggap sama.

```python
df['Sex'] = df['Sex'].str.strip().str.lower()
df['Sex'] = df['Sex'].replace({'m': 'male', 'f': 'female'})
```

---

## 3. Data Preprocessing

Data sudah bersih? Sekarang kita **siapkan** biar model bisa baca! 🔧

### Encoding Kategori

Model ML tidak bisa baca teks, jadi kita perlu convert nilai kategori jadi angka:

- **Label Encoding** → untuk kolom biner (2 nilai). Contoh: `male/female → 0/1`
- **One-Hot Encoding** → untuk kolom dengan lebih dari 2 nilai

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])         # male/female → 0/1

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# → Embarked_Q, Embarked_S (Embarked_C jadi baseline)
```

### Feature Scaling

Banyak algoritma ML sensitif terhadap skala fitur (terutama yang berbasis jarak atau gradient). Ada dua pendekatan utama:

| Metode | Rumus | Hasil |
|--------|-------|-------|
| Standardization | $z = \frac{x - \mu}{\sigma}$ | Mean=0, Std=1 |
| Normalization (MinMax) | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Range [0, 1] |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_std = StandardScaler()
X_standardized = scaler_std.fit_transform(X)

scaler_mm = MinMaxScaler()
X_normalized = scaler_mm.fit_transform(X)
```

### Feature Engineering

Bikin fitur baru dari fitur yang sudah ada untuk memberikan informasi tambahan ke model. Contoh di dataset Titanic:

```python
df['FamilySize']   = df['SibSp'] + df['Parch'] + 1
df['IsAlone']      = (df['FamilySize'] == 1).astype(int)
df['FarePerPerson'] = df['Fare'] / df['FamilySize']
```

### Feature Selection

Pilih fitur yang paling berpengaruh terhadap target. Fitur yang tidak relevan bisa nambahin noise ke model.

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X, y)
```

### Transformasi (Log & Box-Cox)

Untuk kolom yang distribusinya **skewed** (miring), transformasi bisa bikin distribusi lebih mendekati normal:

- **Log Transform** → `np.log1p(x)` — aman untuk nilai 0
- **Box-Cox** → lebih fleksibel, tapi nilai harus > 0

```python
df['Fare_log']   = np.log1p(df['Fare'])
df['Age_boxcox'], _ = stats.boxcox(df['Age'] + 1)
```

---

## 4. Train/Test Split

Terakhir, kita bagi data jadi dua bagian: **training set** dan **testing set**.

> **Analogi:** Training set = materi belajar. Test set = soal ujian yang belum pernah dilihat model. Kalau ujiannya pakai soal yang sama kayak materi, hasilnya gak valid dong! 😄

### Split Biasa (80:20)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Stratified Split

Kapan pakai ini? Kalau dataset kamu **imbalanced**! Stratified split menjamin distribusi kelas tetap proporsional di train dan test set.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### SMOTE (Handling Imbalanced Data)

Kalau kelas tidak seimbang, kita bisa oversample kelas minoritas dengan **SMOTE** (Synthetic Minority Over-sampling Technique).

> ⚠️ **Penting:** SMOTE **hanya** diterapkan ke **training set**! Jangan SMOTE test set-nya karena bisa bocor informasi.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

### Cross Validation

Daripada split sekali, lebih baik pakai **K-Fold Cross Validation** untuk evaluasi yang lebih robust dan tidak bergantung pada satu split tertentu.

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
```

### Stratified K-Fold

Kombinasi stratified + k-fold — distribusi kelas tetap proporsional di setiap fold.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_f_train, X_f_val = X.iloc[train_idx], X.iloc[val_idx]
    y_f_train, y_f_val = y.iloc[train_idx], y.iloc[val_idx]
```

---

### Kesimpulan

Sebelum model dilatih, data perlu melewati 4 tahapan penting:

| Tahap | Tujuan |
|-------|--------|
| **EDA** | Memahami karakteristik dan kondisi data |
| **Data Cleaning** | Membersihkan data dari noise, missing value, dan outlier |
| **Data Preprocessing** | Menyiapkan data agar bisa dibaca model (encoding, scaling, dll.) |
| **Train/Test Split** | Membagi data untuk training dan evaluasi yang jujur |

Kalau semua langkah ini dilakukan dengan benar, model yang dilatih akan jauh lebih akurat dan reliable! 🎯
