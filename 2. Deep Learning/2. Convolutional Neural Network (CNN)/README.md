# Convolutional Neural Network (CNN)

## Daftar Isi

- [Daftar Isi](#daftar-isi)
- [Pengenalan](#pengenalan)
- [Arsitektur CNN](#arsitektur-cnn)
- [Referensi](#referensi)

## Pengenalan

Oke, kita sudah paham dasar-dasar Neural Network. Sekarang, gimana kalau inputnya bukan data tabular biasa, tapi **gambar**? 🖼️

**Convolutional Neural Network (CNN)** adalah arsitektur deep learning yang dirancang khusus untuk memproses data berbentuk **grid** — terutama gambar. CNN menggunakan operasi **konvolusi** untuk mengekstraksi fitur secara hierarkis: dari fitur sederhana (tepi, tekstur) hingga fitur kompleks (wajah, objek).

---

### Analogi gampangnya:

Bayangin kamu melihat foto seekor kucing. Mata kamu nggak langsung bilang "itu kucing!" Pertama, otak kamu mendeteksi **garis-garis**, lalu **bentuk telinga**, lalu **wajah kucing**, baru akhirnya menyimpulkan itu kucing. CNN bekerja persis seperti itu — memproses gambar **lapis demi lapis** dari fitur sederhana ke kompleks! 🐱

---

### Kenapa Neural Network Biasa Tidak Cukup untuk Gambar?

| Masalah | Penjelasan |
|---|---|
| **Terlalu banyak parameter** | Gambar 224×224×3 = 150.528 input neuron. Fully connected → jutaan parameter! |
| **Kehilangan informasi spasial** | MLP memperlakukan setiap piksel secara independen, tanpa memahami posisi relatif |
| **Tidak invariant terhadap translasi** | Objek di posisi berbeda dianggap hal yang berbeda |

CNN mengatasi semua masalah ini dengan **parameter sharing** dan **local connectivity**. 💡

---

## Arsitektur CNN

Berikut adalah topik-topik utama yang akan dipelajari di subbab ini:

| Topik | Deskripsi |
|---|---|
| **Convolution & Pooling** | Operasi konvolusi, stride, padding, max/avg pooling |
| **Arsitektur Populer** | LeNet, AlexNet, VGGNet, ResNet, Inception, EfficientNet |
| **Image Classification** | Klasifikasi gambar end-to-end menggunakan CNN |
| **Object Detection** | YOLO, SSD, Faster R-CNN untuk deteksi objek |
| **Transfer Learning** | Memanfaatkan model pre-trained untuk task baru |

Klik untuk baca lebih lanjut:
- [Convolution & Pooling](ConvPooling/ConvPooling.md)
- [Arsitektur Populer](Arsitektur/Arsitektur.md)
- [Image Classification](ImageClassification/ImageClassification.md)
- [Object Detection](ObjectDetection/ObjectDetection.md)
- [Transfer Learning](TransferLearning/TransferLearning.md)

---

## Implementasi

| File | Deskripsi |
|---|---|
| [Colab Demo CNN](https://colab.research.google.com/drive/1F3Ph5lPohx21OSF_RdPBDyU5N2PCfLLE?usp=sharing) | Demo CNN vs MLP: klasifikasi MNIST, visualisasi feature maps & filter, perbandingan akurasi head-to-head |

---

## Referensi

- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
- [Deep Learning Book - Ian Goodfellow (Chapter 9)](https://www.deeplearningbook.org/contents/convnets.html)
- [PyTorch: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
