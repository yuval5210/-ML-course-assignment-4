# Assignment 4 — Transfer Learning on Oxford Flowers 102 (VGG19 + YOLOv5)

This repository contains the code and experiments for **Assignment 4**: training and evaluating **two pretrained CNN models** (**VGG19** and **YOLOv5 classification**) on the **Oxford Flowers 102** dataset using **transfer learning** in **PyTorch**.  
We run the full pipeline on **multiple stratified random splits** and report learning curves (**accuracy** and **cross-entropy loss**) for **train/validation/test**.

---

## Authors
- **Maidad Maissel** — *Ben-Gurion University of the Negev*
- **Yuval Cohen** — *Ben-Gurion University of the Negev*

---

## Project Overview
All work is implemented and demonstrated in:

- `Assignment_4_ML.ipynb` — dataset download + preprocessing, stratified splits, training loops, evaluation, learning curves, and results summary.

---

## Dataset and Labels
- **Dataset:** Oxford Flowers 102 (102 classes)
- **Files:** image archive + labels in MATLAB `.mat` format
- **Label indexing:** original labels are **1–102** → converted to **0–101** (PyTorch convention), and we build a mapping from image filename/index to class label.

---

## Data Splitting Protocol
### Stratified split (50% / 25% / 25%)
For each experimental run, we create a **random stratified split** to keep class proportions similar across subsets:

1) **Train (50%)** + **Temporary (50%)**  
2) Split Temporary into:
   - **Validation (25%)**
   - **Test (25%)**

> Stratification is important with 102 classes to avoid under-representing some classes in validation/test.

### Repeated splits (multiple random seeds)
We repeat the entire pipeline across **3 different random seeds** (3 independent splits).  
For each split, we train and evaluate **both models** (VGG19 and YOLOv5 classifier), and store learning curves and final test accuracy.

---

## Preprocessing & Data Augmentation
Because both backbones are pretrained on ImageNet-style data, we ensure consistent inputs:
- **RGB** images (3 channels)
- Fixed input size: **224×224**
- **ImageNet normalization** (mean/std)

### Training transforms (augmentation + normalization)
- Random resized crop to 224
- Random horizontal flip
- Random rotation
- Color jitter
- Convert to tensor
- Normalize with ImageNet mean/std

### Validation/Test transforms (deterministic + normalization)
- Resize to a slightly larger size (e.g., 256)
- Center crop to 224
- Convert to tensor
- Normalize with ImageNet mean/std

---

## Models Implemented
### 1) VGG19 (Transfer Learning)
- **Base model:** VGG19 pretrained on ImageNet
- **Strategy:** freeze convolutional feature extractor
- **Head replacement:** replace final classifier layer to output **102 logits**
- **Trainable parameters:** classification head (and any unfrozen classifier layers)

### 2) YOLOv5 Classification (Transfer Learning)
- **Base model:** YOLOv5 **classification** pretrained weights (not detection)
- **Strategy:** freeze backbone
- **Head replacement:** replace final linear layer to output **102 logits**
- **Trainable parameters:** only the new classification head

---

## Training Methodology
- **Loss:** Cross-Entropy Loss (multi-class classification)
- **Metric:** Accuracy (argmax over logits vs true label)
- **Optimizer:** SGD with momentum
- **Batch size:** ~32
- **Epochs:** 25 per split per model

### Probabilistic output
Both models produce **logits**. We convert logits to probabilities using `softmax(logits)` to obtain a probability distribution over the 102 classes.

---

## Results
The assignment requires **≥70%** test accuracy with at least one model. Across repeated stratified splits, both models achieve **~86%** test accuracy consistently.

| Model | Split 1 (Seed 42) | Split 2 (Seed 123) | Split 3 (Seed 1337) |
|---|---:|---:|---:|
| VGG19 | 86.42% | 86.03% | 87.30% |
| YOLOv5 (classification) | 86.42% | 85.83% | 86.32% |

We also generate learning curves (accuracy and cross-entropy loss vs epochs) for train/validation/test for each model and split.
