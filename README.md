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

Before starting training, make sure to split the dataset (see the label_split folder).

---

## 🧠 Training Options

We provide **four training strategies**, and each has separate code for **2-class, 6-class, and 7-class** classification.

You can run each Python script.

---

### 1. Supervised Learning

```
python supervised_code.py      --train_root_dir
                               --test_dir
                               --batch_size 32
                               --num_workers 24
                               --lr 1e-4
                               --epochs 100
                               --patience 10
                               --num_class [2,6,7]
```

---

### 2. Semi-Supervised Learning (Pseudo-labeling)

Using this code, if you only set num_class and do not specify rounds, samples per round and threshold, the default settings will be applied.

```
python Pseudo_labeling_code.py --num_class [2,6,7]
                               --patience 10
                               --rounds 21
                               --samples_per_round 6000
                               --threshold 0.95
                               --batch_size 32
                               --num_workers 24
                               --gpus 0,1,2,3 (default 0)
                               --train_dir
                               --val_dir
                               --test_dir
                               --unlabeled_dir
                               --save_dir
```

---

### 3. Active Learning - BALD  (Bayesian Active Learning by Disagreement) & AD-BALD (Accuracy Driven-BALD)

Using this code, if you only set num_class and do not specify random_sample, topk, or threshold, the default settings will be applied.

#### BALD
```
python Active_learning_code.py --num_class [2,6,7]
                               --method BALD
                               --patience 10
                               --rounds 21
                               --random_sample 6000
                               --batch_size 32
                               --topk 2000
                               --num_workers 24
                               --gpus [1,2,3,4]
                               --train_dir 
                               --val_dir
                               --test_dir
                               --unlabeled_dir
```
#### AD-BALD
```
python Active_learning_code.py --num_class [2,6,7]
                               --method AD-BALD
                               --patience 10
                               --rounds 21
                               --random_sample 6000
                               --threshold 0.5
                               --batch_size 32
                               --num_workers 24
                               --gpus [1,2,3,4]
                               --train_dir 
                               --val_dir
                               --test_dir
                               --unlabeled_dir
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

