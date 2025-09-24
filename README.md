# 📦 InfoColon

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

You can either use the bash scripts (recommended) or run each Python script manually.

---

### 1. Supervised Learning

**Bash (recommended):**

```bash
bash Supervised_learning.sh
```

**Manual:**

```bash
python Supervised_2class.py
python Supervised_6class.py
python Supervised_7class.py
```

---

### 2. Semi-Supervised Learning (Pseudo-labeling)

**Bash (recommended):**

```bash
bash Semi_supervised_learning.sh
```

**Manual:**

```bash
python Semi_supervised_2class.py
python Semi_supervised_6class.py
python Semi_supervised_7class.py
```

---

### 3. Active Learning - BALD  
*(Bayesian Active Learning by Disagreement)*

**Bash (recommended):**

```bash
bash Active_learning_BALD.sh
```

**Manual:**

```bash
python BALD_2class.py
python BALD_6class.py
python BALD_7class.py
```

---

### 4. Active Learning - AD-BALD  
*(Adaptive Threshold BALD - our proposed method)*

**Bash (recommended):**

```bash
bash Active_learning_ADBALD.sh
```

**Manual:**

```bash
python ADBALD_2class.py
python ADBALD_6class.py
python ADBALD_7class.py
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

