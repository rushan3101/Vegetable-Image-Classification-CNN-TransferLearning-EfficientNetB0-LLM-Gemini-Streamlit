# 🥦 Vegetable Image Detection using Deep Learning

## 📘 Project Overview

This project focuses on building a **Deep Learning-based image classification model** capable of identifying different types of vegetables from image data. The goal was to develop an accurate and efficient model using **Convolutional Neural Networks (CNNs)** and **Transfer Learning** techniques to automate vegetable recognition for potential use in retail or inventory systems.

---

## 🎯 Objectives

* Classify vegetable images accurately using deep learning models.
* Experiment with custom CNN architectures and transfer learning to improve performance.
* Optimize the model through fine-tuning to achieve near-perfect accuracy.

---

## 🧠 Approach

1. **Data Preprocessing:**

   * Resized and normalized all input images for consistency.
   * Applied image augmentation to improve model generalization.

2. **Model Development:**

   * Built a baseline **CNN model** achieving ~94% accuracy.
   * Implemented **Transfer Learning** using multiple pre-trained architectures (VGG19, MobileNetV2, EfficientNetB0).
   * Fine-tuned layers and hyperparameters for better feature extraction and generalization.

3. **Model Evaluation:**

   * Achieved **99.9% accuracy** with **Fine-Tuned EfficientNet**, outperforming other architectures.
   * Used accuracy, loss curves, and confusion matrix to evaluate model performance.

---

## 🧩 Tech Stack

* **Language:** Python
* **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, scikit-learn
* **Model Architectures:** CNN, VGG16, MobileNetV2, EfficientNet
* **Tools:** Jupyter Notebook, Google Colab

---

## 📊 Results

| Model                     | Accuracy  | Notes                   |
| ------------------------- | --------- | ----------------------- |
| Custom CNN                | 94.4%     | Baseline model          |
| MobileNetV2               | 99.6%     | Improved generalization |
| VGG19                     | 99.6%     | Improved generalization |
| EfficientNetB0            | **99.9%** | Final best model        |

---

## 🚀 Key Features

* End-to-end deep learning workflow (data preprocessing → model training → evaluation)
* Transfer learning with multiple pre-trained models
* Visualization of training metrics and prediction results
* High accuracy suitable for real-world classification tasks

---

## 🏁 Conclusion

This project demonstrates the power of **Transfer Learning** and **Fine-Tuning** in achieving near-perfect performance on an image classification problem. It can be extended for broader agricultural or retail use cases involving image-based product identification.
