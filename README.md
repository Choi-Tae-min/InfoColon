# InfoColon

This repository contains the classification dataset and related utilities for the InfoColon project.

## Table of Contents
1. [Requirements](#Requirements)
2. [Dataset](#Dataset)
3. [Usage](#Usage)

## Requirements
- We recommend using a machine with **two GPUs**.
- Install required pip packages with their **latest** versions.
- pip install timm pandas torch torchvision matplotlib scikit-learn

## Dataset
Details about the dataset will be provided here.

## Usage
if you download all data or specific data,
python data_split.py

There is 4 option to training
1. Supervised learning
if you can bash file then,
bash Superviesed_learning.sh
but you can't then,
python Supervised_2class.py
python Supervised_6class.py
python Superviesed_7class.py
2. Semi-supervised learning(Pseudo-labeling)
if you can bash file then,
bash Semi_superviesed_learning.sh
but you can't then,
python Semi_supervised_2class.py
python Semi_supervised_6class.py
python Semi_superviesed_7class.py
3. Active learning(BALD, Baseyian Active Learning by Disagreement)
if you can bash file then,
bash Active_learning_BALD.sh
but you can't then,
python BALD_2class.py
python BALD_6class.py
python BALD_7class.py
4. Active learning(AD-BALD, Adaptive threshold BALD)
if you can bash file then,
bash Active_learning_ADBALD.sh
but you can't then,
python ADBALD_2class.py
python ADBALD_6class.py
python ADBALD_7class.py

if you only inference data,
python inference.py
