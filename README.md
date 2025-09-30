# 📦 InfoColon - update complete soon

This repository contains the classification dataset and related training utilities for the **InfoColon** project, designed for informative frame classification in colonoscopy.
Please note that the code may undergo further revisions in the future.

---

## 📚 Table of Contents

1. [Requirements](#requirements)
2. [Dataset Structure](#dataset-structure)
3. [Usage](#usage)
    - [Data Split](#data-split)
    - [Training Options](#training-options)
    - [Testing](#testing)

---

## ⚙️ Requirements

- We recommend using a machine with **two GPUs** for efficient training.
- Install all necessary packages using:

```bash
pip install timm pandas torch torchvision matplotlib scikit-learn
```

---

## 📂 Dataset Structure

Detailed information about the dataset structure and labels will be provided here.

---

## 🚀 Usage

### 🔹 Data Split

Before training, split the dataset using:

```bash
python data_split.py
```

---

## 🧠 Training Options

We provide **four training strategies**, and each has separate code for **2-class, 6-class, and 7-class** classification.

You can run each Python script.

---

### 1. Supervised Learning

```
python supervised_code.py
```

---

### 2. Semi-Supervised Learning (Pseudo-labeling)

```
python Pseudo_labeling_code.py
```

---

### 3. Active Learning - BALD  (Bayesian Active Learning by Disagreement) & AD-BALD (Accuracy Driven-BALD)

```
python Active_learning_code.py
```

---

## 🧪 Testing

> **Note:** Training and test code are implemented **separately** for flexibility.

To evaluate on the test dataset, use:

```bash
python inference.py
```

This will load the trained model and generate performance metrics on the test set.

---

Feel free to open an issue if you encounter any problems!

