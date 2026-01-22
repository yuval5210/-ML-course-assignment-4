# Assignment 3 — Deep Neural Network on MNIST

This repository contains the code and experiments for **Assignment 3**: extending the Chapter 11 scratch MLP to **two hidden layers**, and comparing it with a **PyTorch fully-connected neural network** on **MNIST** dataset.

## Authors
- **Maidad Maissel** — *Ben-Gurion University of the Negev* 
- **Yuval Cohen** — *Ben-Gurion University of the Negev*

## Project Overview
All the work is implemented and demonstrated in:

- `ch11_new.ipynb` — scratch MLP (1 hidden) + scratch MLP (2 hidden) + PyTorch FCNN (2 hidden), training curves, and evaluation.

## Dataset and split
- **Dataset:** MNIST (`mnist_784`) loaded via `fetch_openml`
- **Preprocessing:** pixel normalization to **[-1, 1]**
- **Split used in the notebook:**
  - **Test:** 30% (21,000 samples)  
  - Remaining 70% is split again into:
    - **Train:** 34,300 samples  
    - **Validation:** 14,700 samples  

> Note: The assignment requires Train(70%)/Test(30%). 
> In the notebook we still keep a **validation** set (from the 70%) to monitor training and select the best PyTorch epoch.

## Models implemented

### 1) Scratch MLP (1 hidden layer — Chapter 11 baseline)
- **Architecture:** 784 → 500 → 10
- **Activations:** Sigmoid in hidden + Sigmoid at output (matches the Chapter 11 style)
- **Loss used for training/monitoring:** MSE (with one-hot targets)
- **Metrics:** Accuracy + Macro AUC (OvR)

### 2) Scratch MLP (2 hidden layers — required extension)
- **Architecture:** 784 → 500 → 500 → 10
- **Activations:** Sigmoid (hidden1) → Sigmoid (hidden2) → **Softmax (output)**
- **Stability:** Sigmoid uses clipping (`np.clip`) for numerical stability
- **Loss used in backward:** MSE-style gradient + **Softmax Jacobian** for correct multi-class probabilities
- **Metrics:** Accuracy + Macro AUC (OvR)

### 3) PyTorch FCNN (2 hidden layers — framework comparison)
- **Architecture:** 784 → 500 → 500 → 10
- **Activations (as implemented):** Sigmoid after each layer (including output)
- **Loss:** `nn.MSELoss()` with one-hot targets
- **Optimization:** SGD (lr=0.1), minibatch size 100
- **Model selection:** best epoch by validation Macro AUC

## Evaluation metrics
- **Accuracy** (argmax over class scores/probabilities)
- **Macro AUC (One-vs-Rest)** — multi-class ROC-AUC with `average="macro"` (as requested in the assignment).

## Results (as printed by `ch11_new.ipynb`)
These are the final **test** metrics printed in the notebook outputs:

| Model | Test MSE | Test Accuracy | Test Macro AUC |
|---|---:|---:|---:|
| Scratch 1-hidden (784→500→10) | 0.01 | 95.66% | 0.9961 |
| Scratch 2-hidden (784→500→500→10) | 0.01 | 91.08% | 0.9915 |
| PyTorch FCNN 2-hidden (784→500→500→10) | 0.06 | 62.38% | 0.9472 |

## Installation & Requirements
To run the notebook locally, ensure you have the following dependencies installed:
   ```bash
   pip install numpy matplotlib scikit-learn torch
   ```