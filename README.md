
# Deep Learning for Pulmonary Diagnostics: From Classification to Generative Modeling

## 📌 Project Overview

This repository contains a comprehensive deep learning pipeline for analyzing medical lung imagery. The project covers the full lifecycle of medical AI, starting from raw image classification using **Custom CNNs** and **Transfer Learning**, to advanced generative tasks using **GANs** and **AutoEncoders**.

The core objective is to differentiate between **Normal** and **Pneumonia** cases while exploring how synthetic data generation and dimensionality reduction can assist in medical contexts.

---

## 🛠 Features & Tasks

### 1. Image Classification Suite

* **Task 1: Custom CNN Architecture**: A baseline model built from scratch to establish performance benchmarks for medical image classification.
* **Task 2: Transfer Learning (Feature Extraction)**: Implementing **EfficientNet-B7**. By freezing the backbone and training only the classification head, we leverage pre-trained ImageNet weights for medical features.
* **Task 3: Fine-Tuning**: Unfreezing the top 33% of the EfficientNet-B7 layers to specialize the model specifically for the nuances of X-ray imagery.

### 2. Generative & Reconstructive Models

* **Task 4: Generative Adversarial Networks (GANs)**: Training a Generator and Discriminator to synthesize artificial pneumonia lung images (32x32 resolution) from latent noise.
* **Task 5: AutoEncoder Reconstruction**: Developing an unsupervised model to compress images into a low-dimensional bottleneck and reconstruct them, useful for denoising and feature extraction.

---

## 📊 Dataset Information

* **Domain**: Pediatric Pneumonia Chest X-ray images.
* **Structure**: 5,856 total images organized into Training, Validation, and Testing sets.
* **Pre-processing**:
* Resizing (256x256 for CNN/AutoEncoder; 32x32 for GAN).
* Class weighting (to handle the ~3:1 imbalance between Pneumonia and Normal cases).
* Data Augmentation: Random rotations, horizontal flips, and color jittering.



---

## 🚀 Technical Implementation

### Model Optimization

* **Loss Functions**: `BCEWithLogitsLoss` for classification (with weights) and `MSELoss` for the AutoEncoder.
* **Optimizer**: Stochastic Gradient Descent (SGD) with Momentum (0.9) and Adam for generative tasks.
* **Learning Rate**: Dynamic scheduling using `StepLR`.

### Evaluation Metrics

We measure performance through multiple lenses to ensure clinical relevance:

* **Accuracy & Loss Curves**
* **Precision, Recall, & F1-Score** (Crucial for medical diagnostics)
* **Confusion Matrix Analysis**
* **Visual Validation** for GAN and AutoEncoder outputs.

---

## 📈 Results Summary

| Model | Key Outcome |
| --- | --- |
| **Custom CNN** | Established solid baseline performance with high interpretability. |
| **EfficientNet-B7 (TL)** | High precision through state-of-the-art feature extraction. |
| **Fine-Tuning** | Significant boost in Recall, minimizing false negatives. |
| **AutoEncoder** | High structural similarity index (SSIM) in reconstructions. |
| **GAN** | Successfully generated synthetic X-ray patterns from noise. |

---

## ⚙️ Setup & Installation

1. **Clone the Repository**:
```bash
git clone https://github.com/your-username/medical-dl-project.git
cd medical-dl-project

```


2. **Install Dependencies**:
```bash
pip install torch torchvision torchmetrics matplotlib scikit-learn

```


3. **Hardware Requirements**:
* Recommended: NVIDIA T4 GPU or better (CUDA enabled).
* Estimated Training Time: ~2-3 hours for the full pipeline.



---

## 👥 Authors

* **Shayan Rokhva** - *Tarbiat Modares University*
* **Ali Rashedi** - *Tarbiat Modares University*
* **Supervised by**: Dr. Khatibi

---

*This project was completed as part of the Deep Learning Course (1403-1404 Academic Year) at Tarbiat Modares University, Department of IT Engineering.*
