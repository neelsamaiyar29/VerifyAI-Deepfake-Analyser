# DeepFake Detection Models

This repository contains code on detecting DeepFakes using a hybrid CNN–LSTM architecture. The goal is to distinguish real from fake videos by extracting spatial features with ResNeXt‑50 and modeling temporal dynamics with LSTM layers.

We train and evaluate the same model on two datasets:

1. **DFDC (DeepFake Detection Challenge)**
2. **AvLips (LipSync) / LipSync Dataset**

---

---

## 📈 Performance

| Dataset    | Train Accuracy | Test Accuracy |
|:----------:|:--------------:|:-------------:|
| **DFDC**   |     96.47 %     |    95.19 %    |
| **AvLips** |     98.15 %     |    98.88 %    |

---

## 📋 Table of Contents

* [🚀 Overview](#-overview)
* [📂 Repository Structure](#-repository-structure)
* [⚙️ Environment Setup](#️-environment-setup)
* [📥 Data Preparation](#-data-preparation)

  * [DFDC Dataset](#dfdc-dataset)
  * [AvLips Dataset](#avlips-dataset)
* [🏃‍♂️ Training Instructions](#-training-instructions)

  * [Train on DFDC](#train-on-dfdc)
  * [Train on AvLips](#train-on-avlips)
* [📊 Evaluation & Metrics](#-evaluation--metrics)
* [📈 Results Visualization](#-results-visualization)
* [🛠️ Dependencies](#️-dependencies)
* [🤝 Contributing](#-contributing)
* [📝 License](#-license)

---

## 🚀 Overview

As part of our undergraduate capstone, We built a DeepFake detection pipeline. It:

* Samples and preprocesses video frames uniformly
* Extracts frame-level features using a pre-trained ResNeXt‑50 backbone
* Models frame sequences with an LSTM
* Outputs binary predictions (real vs. fake)
* Reports accuracy, precision, recall, F1-score, and confusion matrices

---

## 📂 Repository Structure

```bash
├── data/                 
│   ├── dfdc_dataset/          
│   └── avlips_dataset/       
├── deepfake_detection_dfdc.ipynb   
├── deepfake_detection_AvLips.ipynb   
├── requirements.txt           
└── README.md                
```

---

## ⚙️ Environment Setup

1. Clone this repo:

   ```bash
   git clone https://github.com/sangeetanandanvishal04/deepfake_detection.git
   cd deepfake_detection
   ```
2. Install Python packages:

   ```bash
   pip install -r requirements.txt
   ```

> **Note**: For GPU training on Colab, use the CUDA build:
>
> ```bash
> pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
> ```

---

## 📥 Data Preparation

### DFDC Dataset

* The notebook downloads `dfdc_dataset.zip` via `gdown` and extracts into `data/dfdc_dataset/real` and `.../fake`.

### AvLips Dataset

* Provide your AvLips archive’s Google Drive file ID in the notebook.
* The same download-and-extract flow populates `data/avlips_dataset/real` and `.../fake`.

---

## 🏃‍♂️ Training Instructions

Adjust hyperparameters at the top of the notebook:

```python
BATCH_SIZE = 8
NUM_EPOCHS = 10
learning_rate = 1e-5
weight_decay = 1e-6
```

### Train on DFDC

1. Ensure `DATASET_FOLDER = Path("data/dfdc_dataset")`.
2. Run all cells to preprocess, define model, and train.

### Train on AvLips

1. Change to `DATASET_FOLDER = Path("data/avlips_dataset")`.
2. Run preprocessing and training cells again.

---

## 📊 Evaluation & Metrics

After training, the notebook computes these metrics on both train/test splits:

* Loss (training vs. test)
* Accuracy
* Precision
* Recall
* F1‑score
* Confusion matrix

---

## 📈 Results Visualization

Loss and accuracy curves are plotted using `matplotlib`. You can extend this to TensorBoard or other visualization tools.

---

## 🛠️ Dependencies

See `requirements.txt`:

```
torch
torchvision
gdown
matplotlib
scikit-learn
opencv-python
torchinfo
tqdm
Pillow
```

---

## 🤝 Contributing

As a fellow student or researcher, your feedback is welcome! Feel free to open issues or pull requests.

---