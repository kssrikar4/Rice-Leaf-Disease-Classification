# Rice Leaf Disease Classification using Swin Transformers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/kssrikar4/Rice-Leaf-Disease-Classification)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains a deep learning implementation for the automated detection and classification of rice leaf diseases. Early detection of plant diseases is critical for maintaining crop yield and ensuring food security. 

This project leverages **Transfer Learning** using the **Swin Transformer (Tiny)** architecture, a hierarchical Vision Transformer that has shown superior performance in image classification tasks compared to traditional CNNs. The model is trained on a consolidated dataset of rice leaf images to classify various pathological conditions.

## Key Features

*   **State-of-the-Art Architecture:** Utilizes `swin_tiny_patch4_window7_224` from the `timm` library.
*   **Advanced Training Strategy:**
    *   **Optimizer:** AdamW with weight decay.
    *   **Scheduler:** OneCycleLR with 5% warmup and cosine annealing.
    *   **Regularization:** Label smoothing (0.1) and heavy data augmentation (Albumentations).
    *   **Performance:** Mixed Precision Training (AMP) for faster computation and lower memory usage.
*   **High Accuracy:** Achieved **~97.1% Validation Accuracy** within 15 epochs.

## Dataset

The dataset is a consolidation of three public datasets from Kaggle, preprocessed into a standard ImageFolder structure:

1.  [Rice Disease Dataset](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)
2.  [Rice Leaf Disease Image](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image)
3.  [Rice Leaf Diseases](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)

**Classes Identified:**
*   Bacterial Leaf Blight
*   Brown Spot
*   Leaf Blast
*   Sheath Blight
*   Tungro
*   Leaf Scald
*   Healthy (if present in the specific split used)

## Results

The model demonstrates rapid convergence and high generalization capabilities.

*   **Final Validation Accuracy:** 97.10%
*   **Final Training Loss:** ~0.50

**Training Metrics:**
<img width="720" height="360" alt="metrics" src="https://github.com/user-attachments/assets/9f8ba677-4514-4dda-a32c-97c21ded6052" />

**Sample Predictions:**
The grid below displays model predictions on the validation set. Green text indicates correct classifications.
<img width="864" height="864" alt="predictions" src="https://github.com/user-attachments/assets/07462839-d112-4e2e-ab97-70c1697b7763" />

**Configuration:**
*   `BATCH_SIZE`: Default 16
*   `MAX_LR`: Default 1e-4
*   `EPOCHS`: Default 15

## Download

This model is hosted on the Hugging Face Model Hub and can be easily integrated into your projects.

### Model Hub Link
🔗 [kssrikar4/Rice-Leaf-Disease-Classification](https://huggingface.co/kssrikar4/Rice-Leaf-Disease-Classification)


## Project Structure

| File | Description |
| :--- | :--- |
| `main.py` | Core script containing the model definition, training loop, and inference logic. |
| `knowdata.ipynb` | Jupyter notebook used for initial data exploration, unzipping archives, and verifying class distributions. |
| `history.json` | JSON log containing epoch-wise loss and accuracy metrics. |

## License

This project is licensed under the MIT License.

## Acknowledgments

*   **Ross Wightman** for the `timm` (PyTorch Image Models) library.
*   **Kaggle Contributors** for providing the open-source datasets.
*   **Google Gemini** For assistance in code generation and analysis.
