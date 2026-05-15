# Neural Network Fundamentals

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Pengenalan](#pengenalan)
- [Komponen Neural Network](#komponen-neural-network)
- [Referensi](#referensi)

## Pengenalan

Sebelum kita masuk ke arsitektur deep learning yang lebih kompleks seperti CNN, RNN, atau Transformer, kita perlu memahami dulu **fondasi** dari semua itu: **Neural Network**.

![image neural net](https://cdn.the-scientist.com/assets/articleNo/71687/aImg/52292/62dc0501-8dda-4bd7-9ba9-fa1a9b8c7cb4-l.webp)

Neural Network (atau *Artificial Neural Network*) adalah model komputasi yang terinspirasi dari cara kerja **neuron di otak manusia**. Terdiri dari unit-unit kecil (neuron) yang saling terhubung dan memproses informasi secara berlapis-lapis.

---
### Neuron Otak Manusia vs Neuron Buatan

![image neuron](https://dicoding-assets.sgp1.cdn.digitaloceanspaces.com/blog/wp-content/uploads/2024/07/BLOG-Aset-3.jpg)

lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

---

### Analogi gampangnya:

Bayangin kamu punya tim yang bekerja berantai. Orang pertama menerima data mentah, memproses sedikit, lalu passing hasilnya ke orang berikutnya. Setiap orang menambahkan "pemahaman" sedikit demi sedikit, sampai orang terakhir bisa memberikan jawaban akhir. Itulah cara kerja Neural Network! 

---

### Perbedaan Machine Learning vs Deep Learning

| | Machine Learning | Deep Learning |
|---|---|---|
| **Arsitektur** | Algoritma tradisional (tree, SVM, dll) | Neural Network berlapis-lapis |
| **Feature Engineering** | Manual (harus dilakukan manusia) | Otomatis (dipelajari oleh model) |
| **Data yang dibutuhkan** | Bisa sedikit | Butuh banyak data |
| **Komputasi** | CPU cukup | Butuh GPU/TPU |
| **Cocok untuk** | Data tabular, structured | Gambar, teks, audio, video |

---

## Komponen Neural Network

Berikut adalah komponen-komponen utama yang akan dipelajari di subbab ini:

| Topik | Deskripsi |
|---|---|
| **Perceptron & MLP** | Dari neuron tunggal ke jaringan berlapis-lapis |
| **Activation Functions** | Fungsi non-linear yang membuat NN bisa mempelajari pola kompleks |
| **Forward & Backward Propagation** | Bagaimana data mengalir dan bobot diperbarui |
| **Loss Functions** | Mengukur seberapa "salah" prediksi model |
| **Gradient Descent & Variants** | Optimizer untuk meminimalkan loss |

Klik untuk baca lebih lanjut:
- [Perceptron & MLP](Perceptron/Perceptron.md)
- [Activation Functions](ActivationFunctions/ActivationFunctions.md)
- [Forward & Backward Propagation](Propagation/Propagation.md)
- [Loss Functions](LossFunctions/LossFunctions.md)
- [Gradient Descent](GradientDescent/GradientDescent.md)

---

## Implementasi

| File | Deskripsi |
|---|---|
| [Colab ANN Demo](https://colab.research.google.com/drive/1NXhrjtPdRXz0P_vL6YSMUOLTF8QA3dGz?usp=sharing) | Demo lengkap: bangun ANN dari nol (NumPy) lalu dengan PyTorch untuk klasifikasi MNIST |

---

## Referensi

- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Deep Learning Book - Ian Goodfellow (Chapter 6)](https://www.deeplearningbook.org/contents/mlp.html)
- [PyTorch: Neural Network Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- [Dicoding: Neural Network: Cikal Bakal Revolusi Deep Learning](https://www.dicoding.com/blog/neural-network-cikal-bakal-revolusi-deep-learning/)
