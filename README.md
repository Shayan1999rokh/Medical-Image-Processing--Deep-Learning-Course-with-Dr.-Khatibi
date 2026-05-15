# 🫁 Deep Learning for Medical Image Processing

## Pneumonia Detection, Feature Extraction, Fine-Tuning, GAN, and AutoEncoder

A comprehensive deep learning project focused on **medical image analysis of chest X-ray images** for **binary classification of Pneumonia vs Normal lungs**, along with **synthetic image generation using GANs** and **image reconstruction using AutoEncoders**.

This repository was developed as a university deep learning project for the course **Deep Learning** in the field of **Information Systems Engineering**.

---

# 📌 Project Overview

This project implements **five different deep learning tasks**:

| Task   | Description                                                  |
| ------ | ------------------------------------------------------------ |
| Task 1 | Custom CNN from scratch for binary classification            |
| Task 2 | Transfer Learning using EfficientNet-B7 (Feature Extraction) |
| Task 3 | Fine-Tuning EfficientNet-B7                                  |
| Task 4 | GAN for synthetic pneumonia image generation                 |
| Task 5 | Convolutional AutoEncoder for image reconstruction           |

The dataset consists of **Chest X-Ray images** categorized into:

* **NORMAL**
* **PNEUMONIA**

---

# 🧠 Objectives

The primary goals of this project are:

* Detect pneumonia from chest X-ray images
* Compare CNN architectures
* Study transfer learning effectiveness
* Generate synthetic medical images using GANs
* Learn latent representations using AutoEncoders
* Analyze performance using multiple evaluation metrics

---

# 🏛️ Academic Information

* **University:** Tarbiat Modares University
* **Faculty:** Industrial Engineering and Systems
* **Department:** Information Technology
* **Course:** Deep Learning
* **Instructor:** Dr. Khatibi
* **Semester:** Fall 1403 / 2024
* **Authors:** Shayan Rokhva & Ali Rashedi

---

# 📂 Dataset

The dataset contains chest X-ray images for binary classification.

## Dataset Statistics

| Split    | Number of Images |
| -------- | ---------------- |
| Training | 5232             |
| Testing  | 624              |

Training data was further divided into:

* 75% Training
* 25% Validation

### Final Split

| Subset     | Images |
| ---------- | ------ |
| Train      | 3924   |
| Validation | 1308   |
| Test       | 624    |

---

# ⚠️ Class Imbalance

The dataset is imbalanced:

| Class     | Percentage |
| --------- | ---------- |
| PNEUMONIA | ~74%       |
| NORMAL    | ~26%       |

Instead of using:

* Oversampling
* Undersampling

we used a **weighted BCEWithLogitsLoss** to reduce class bias.

---

# 🖼️ Image Preprocessing

Original image sizes ranged approximately from:

* 1500 × 1500
* 2500 × 2500

Due to hardware limitations, all images were resized.

## Image Sizes Used

| Model              | Image Size |
| ------------------ | ---------- |
| CNN / EfficientNet | 256 × 256  |
| GAN                | 32 × 32    |
| AutoEncoder        | 256 × 256  |

---

# 🔄 Data Augmentation

The following augmentations were applied to training data:

```python
Resize
RandomHorizontalFlip
RandomRotation
ColorJitter
ToTensor
Normalize
```

### Why No Random Erasing?

Random Erasing was intentionally avoided because removing parts of lung regions may damage medically relevant information.

---

# 🧮 Normalization

Since transfer learning was used, normalization values from ImageNet were applied:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

# ⚙️ Hardware

Training was performed on:

* GPU with 16GB VRAM
* PyTorch framework

The code automatically supports:

```python
CPU
GPU
```

using:

```python
device = torch.device(...)
```

---

# 🧩 Task 1 — Custom CNN from Scratch

## Architecture

The CNN architecture was inspired by VGG-style networks.

### Components

* Convolution Layers
* Batch Normalization
* ReLU Activation
* MaxPooling
* AdaptiveAvgPooling
* Fully Connected Layer

---

## CNN Pipeline

```text
Input Image
   ↓
Conv + BN + ReLU
   ↓
Conv + BN + ReLU
   ↓
MaxPool
   ↓
Repeat Blocks
   ↓
AdaptiveAvgPool
   ↓
Flatten
   ↓
Linear Layer
   ↓
Binary Output
```

---

## Important Design Choices

### Why BCEWithLogitsLoss?

Instead of:

```python
Sigmoid + BCELoss
```

we used:

```python
BCEWithLogitsLoss
```

because it is:

* More numerically stable
* Faster
* Recommended for binary classification

---

# 📉 Loss Function

Weighted BCE loss was used to handle class imbalance.

Example:

```python
loss_fn = nn.BCEWithLogitsLoss(pos_weight=...)
```

---

# 🚀 Optimizer and Scheduler

## Optimizer

```python
SGD
```

with:

| Hyperparameter | Value |
| -------------- | ----- |
| Learning Rate  | 0.01  |
| Momentum       | 0.9   |
| Weight Decay   | 1e-5  |
| Nesterov       | True  |

---

## Learning Rate Scheduler

```python
StepLR
```

| Parameter | Value |
| --------- | ----- |
| step_size | 5     |
| gamma     | 0.5   |

---

# 📊 Evaluation Metrics

The following metrics were tracked:

* Loss
* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

Implemented using:

```python
torchmetrics
```

---

# 🔁 Training Pipeline

Each epoch included:

1. Training phase
2. Validation phase
3. Scheduler update
4. Metric logging
5. Best model saving

---

# 💾 Best Model Saving

The best model was selected based on:

```python
Highest Validation Accuracy
```

and saved using:

```python
torch.save(...)
```

---

# 📈 Task 1 Results

The custom CNN achieved strong performance on:

* Training set
* Validation set
* Unseen Test set

Evaluation included:

* ROC curves
* Confusion matrices
* Accuracy/Loss plots

---

# 🧠 Task 2 — EfficientNet-B7 Feature Extraction

Transfer learning was implemented using:

EfficientNet

---

## Strategy

* All backbone layers were frozen
* Only classifier head was trainable

---

## Advantages

* Faster convergence
* Lower computational cost
* Better generalization on small datasets

---

## Trainable Parameters

| Type                 | Count  |
| -------------------- | ------ |
| Total Parameters     | ~63.8M |
| Trainable Parameters | ~2.5K  |

---

# 🔥 Task 3 — EfficientNet-B7 Fine-Tuning

In this phase:

* First 66% of layers were frozen
* Last 33% were trainable

This allowed:

* Deep feature adaptation
* Better medical specialization

---

## Comparison

| Model  | Learning Strategy  |
| ------ | ------------------ |
| Task 2 | Feature Extraction |
| Task 3 | Fine-Tuning        |

---

# 🤖 Task 4 — GAN (Generative Adversarial Network)

A simple GAN architecture was implemented to generate synthetic pneumonia images.

---

# 🧱 GAN Architecture

## Generator

Fully connected MLP-based architecture:

```text
Noise Vector
   ↓
Linear Layers
   ↓
Generated Image
```

---

## Discriminator

Binary classifier:

```text
Input Image
   ↓
Linear Layers
   ↓
Real / Fake Prediction
```

---

# ⚠️ GAN Constraints

Due to hardware limitations:

* Images resized to 32×32
* Simple FC-based GAN used
* Training epochs limited to 200

---

# 🎯 GAN Objective

The main objective:

* Reduce Generator Loss
* Fool the Discriminator

---

# 📉 GAN Training Behavior

Expected behavior:

| Component          | Expected Trend |
| ------------------ | -------------- |
| Generator Loss     | Decrease       |
| Discriminator Loss | Increase       |

This behavior was observed during training.

---

# 🖼️ GAN Outputs

Generated images improved gradually across epochs:

* Epoch 1
* Epoch 100
* Epoch 200

Even though outputs were low-resolution, the training process behaved correctly.

---

# 🧬 Task 5 — AutoEncoder

A convolutional AutoEncoder was implemented for:

* Feature learning
* Image reconstruction
* Latent representation learning

---

# 🧱 AutoEncoder Architecture

## Encoder

```text
Conv → ReLU → MaxPool
```

Repeated multiple times to compress the image.

---

## Decoder

```text
UpSampling → Conv → ReLU
```

used to reconstruct the image.

---

# 🎯 Objective

Reconstruct input images with minimum reconstruction error.

---

# 📉 Loss Function

```python
Mean Squared Error (MSE)
```

---

# 🖼️ Reconstruction Results

The reconstructed images were visually very similar to original chest X-rays, indicating successful feature compression and reconstruction.

---

# 📚 Technologies Used

| Technology   | Usage                            |
| ------------ | -------------------------------- |
| Python       | Main programming language        |
| PyTorch      | Deep learning framework          |
| TorchMetrics | Evaluation metrics               |
| torchvision  | Pretrained models and transforms |
| Matplotlib   | Visualization                    |
| NumPy        | Numerical operations             |

---

# 📁 Project Structure

```text
project/
│
├── CNN/
│   ├── model.py
│   ├── train.py
│   └── evaluation.py
│
├── EfficientNet_FeatureExtraction/
│
├── EfficientNet_FineTuning/
│
├── GAN/
│
├── AutoEncoder/
│
├── datasets/
│
├── saved_models/
│
├── outputs/
│
├── figures/
│
└── README.md
```

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Train CNN

```bash
python train_cnn.py
```

---

## Train EfficientNet Feature Extraction

```bash
python train_feature_extraction.py
```

---

## Train Fine-Tuning Model

```bash
python train_finetuning.py
```

---

## Train GAN

```bash
python train_gan.py
```

---

## Train AutoEncoder

```bash
python train_autoencoder.py
```

---

# 📊 Visualizations

This repository includes:

* Accuracy plots
* Loss plots
* Precision plots
* Recall plots
* F1-score plots
* ROC-AUC curves
* Confusion matrices
* GAN generated samples
* AutoEncoder reconstruction comparisons

---

# 🧪 Experimental Notes

## CNN

* Strong baseline performance
* Stable convergence

## EfficientNet-B7

* Significant improvement over CNN
* Faster convergence
* Better generalization

## GAN

* Educational implementation
* Limited by image resolution and hardware

## AutoEncoder

* Successful reconstruction
* Effective latent feature extraction

---

# ⚠️ Limitations

* Limited GPU resources
* GAN used low-resolution images
* Limited training epochs for GAN
* No segmentation/localization tasks
* Binary classification only

---

# 🔮 Future Work

Potential future improvements:

* Use Diffusion Models
* Use DCGAN or StyleGAN
* Higher resolution GAN training
* Multi-class lung disease classification
* Vision Transformers (ViT)
* Explainable AI (Grad-CAM)
* Medical image segmentation

---

# 🙏 Acknowledgements

Special thanks to:

* Dr. Khatibi
* Tarbiat Modares University
* PyTorch community

---

# 📜 License

This project is intended for:

* Educational purposes
* Research purposes

Please cite appropriately if used in academic work.

---

# 📬 Contact

For questions, collaborations, or academic discussions:

* GitHub Issues
* Pull Requests
* Academic communication

---

# ⭐ Final Note

This project was designed not only to achieve strong medical image classification performance, but also to provide a complete educational pipeline covering:

* CNNs
* Transfer Learning
* Fine-Tuning
* GANs
* AutoEncoders

within a single coherent deep learning research project.

* ---
!!! ReadMe was produced using ChatGPT. Check important information again before using !!!


