# 🧠 Silicon Wafer Defect Detection

This project uses machine learning and deep learning techniques to automatically detect defects in silicon wafers. By leveraging **transfer learning with a pre-trained VGG16 model** and **CNN**, the system classifies images into defect categories with over **97% accuracy**.

---

## 🚀 Project Overview

Manual inspection and sensor-based techniques for silicon wafer defect detection are often inaccurate and time-consuming. This project automates the detection process using:

- 🧠 Pre-trained **VGG16** for feature extraction
- 🔁 **CNN with Transfer Learning** for classification
- 📊 Evaluation using **accuracy, precision, recall, F1-score**, and **confusion matrix**

---

## 🗂️ Dataset

We used the publicly available dataset from Kaggle:  
🔗 [Silicon Wafer Map Dataset (WM811K)](https://www.kaggle.com/datasets/muhammedjunayed/wm811k-silicon-wafer-map-dataset-image/data)

This dataset includes:
- 8 different defect types
- 1 "Normal" label (no defect)

---

## 📈 Results

- ✅ **Accuracy**: 97%
- 📊 High scores across **precision**, **recall**, and **F1-score**
- 🧩 Confusion Matrix: High true positive rate, very low misclassifications
- 📉 Training graph shows increasing accuracy and decreasing loss with epochs

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Matplotlib
- Scikit-learn
