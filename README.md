# PlantfyCropNetLight: A Lightweight Multi-Branch Attention Network for Fine-Grained Multi-Crop Pathology Classification

[![Parameters](https://img.shields.io/badge/Parameters-0.139M-brightgreen.svg)]()
[![Datasets](https://img.shields.io/badge/Datasets-TCP%20%7C%20ABCGMP-blue.svg)]()
[![Task](https://img.shields.io/badge/Task-Fine--Grained%20Plant%20Pathology-orange.svg)]()
[![Deployment](https://img.shields.io/badge/Deployment-Edge%20Devices-lightgrey.svg)]()

## Overview

**PlantfyCropNetLight** is an ultra-lightweight, parameter-efficient convolutional neural network (CNN) engineered specifically for robust, real-time plant disease classification on resource-constrained edge devices. 

Deploying deep learning models in real-world agricultural fields is often hindered by high computational overhead, memory constraints, complex field background noise, intra-class variations, and subtle visual similarities between different crop diseases. PlantfyCropNetLight overcomes these challenges with an exceptionally low footprint of just **$0.139$ million parameters** while delivering competitive diagnostic accuracy across multiple horticultural and vegetable pathology benchmarks.

---

## Key Architectural Highlights

* **Multi-Branch Depthwise Separable Convolutions:** Extracts multi-scale lesion features across varying spatial resolutions while keeping parameter count and FLOPs minimal.
* **Dual Global Pooling ($\text{GAP} + \text{GMP}$):** Combines Global Average Pooling and Global Max Pooling to capture both holistic context and salient localized disease indicators for enhanced feature representation.
* **Convolutional Block Attention Modules (CBAM):** Refines channel and spatial feature maps to focus on subtle disease symptoms while actively suppressing noisy field backgrounds.
* **Dataset-Specific Activation Optimization:** Empirical investigation within the shared Multi-Layer Perceptron (MLP) block of CBAM demonstrates that:
  * **Swish** optimizes generalization on the **ABCGMP** dataset.
  * **ReLU** achieves superior performance on the **TCP** dataset.

---

## Experimental Results

Evaluated across the **TCP** and **ABCGMP** multi-crop pathology benchmarks spanning eight major fruit and vegetable crops, PlantfyCropNetLight outperforms leading state-of-the-art lightweight models across Accuracy, Precision, Recall, and $F_1$-Score:

| Dataset | Total Crops | Total Classes | Parameters | CBAM MLP Activation | Training Accuracy | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCP** | $3$ | $20$ | $0.139\text{M}$ | **ReLU** | $94.45\%$ | **$86.80\%$** |
| **ABCGMP** | $6$ | $24$ | $0.139\text{M}$ | **Swish** | $91.88\%$ | **$84.16\%$** |

---

## Datasets

### 1. TCP Dataset
The **TCP** dataset comprises **$9,541$ images** distributed across **3 main crops** (Tomato, Chilli, and Papaya) spanning **$20$ distinct classes**.

<details>
<summary><b>Click to view TCP Class & Image Distribution</b></summary>

| S.No | Crop | Class | No. of Images |
| :---: | :--- | :--- | :---: |
| $1$ | Tomato | Healthy Leaf | $1,378$ |
| $2$ | Tomato | Septoria Leaf | $1,617$ |
| $3$ | Tomato | Verticillium Wilt | $673$ |
| $4$ | Tomato | Disease Tomato | $209$ |
| $5$ | Tomato | Healthy Tomato | $325$ |
| $6$ | Chilli | Bacterial Spot | $256$ |
| $7$ | Chilli | Healthy Leaf | $718$ |
| $8$ | Chilli | Cercospora Leaf Spot | $180$ |
| $9$ | Chilli | Curl Virus | $633$ |
| $10$ | Chilli | Nutrition Deficiency | $444$ |
| $11$ | Chilli | White Spot | $195$ |
| $12$ | Chilli | Disease Chilli | $174$ |
| $13$ | Chilli | Healthy Chilli | $218$ |
| $14$ | Papaya | Anthracnose | $360$ |
| $15$ | Papaya | Bacterial Spot | $456$ |
| $16$ | Papaya | Curl | $420$ |
| $17$ | Papaya | Healthy Leaf | $298$ |
| $18$ | Papaya | Disease Papaya | $127$ |
| $19$ | Papaya | Healthy Papaya | $143$ |
| $20$ | Papaya | Ring Spot | $717$ |
| | **Total** | **$20$ Classes** | **$9,541$** |

</details>

---

### 2. ABCGMP Dataset
The **ABCGMP** dataset comprises **$10,113$ images** distributed across **6 major fruit crops** (Apple, Banana, Citrus, Guava, Mango, and Papaya), with $4$ classes per crop spanning **$24$ distinct classes**.

<details>
<summary><b>Click to view ABCGMP Class & Image Distribution</b></summary>

| S.No | Crop | Class | No. of Images | S.No | Crop | Class | No. of Images |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| $1$ | Apple | Healthy Leaf | $2,008$ | $13$ | Guava | Red Rust | $90$ |
| $2$ | Apple | Black Rot | $1,987$ | $14$ | Guava | Healthy Leaf | $150$ |
| $3$ | Apple | Healthy Fruit | $301$ | $15$ | Guava | Disease Fruit | $160$ |
| $4$ | Apple | Disease Fruit | $286$ | $16$ | Guava | Healthy Fruit | $50$ |
| $5$ | Banana | Cordana | $739$ | $17$ | Mango | Bacterial Canker | $404$ |
| $6$ | Banana | Healthy Leaf | $535$ | $18$ | Mango | Healthy Leaf | $308$ |
| $7$ | Banana | Healthy Fruit | $133$ | $19$ | Mango | Disease Fruit | $213$ |
| $8$ | Banana | Disease Fruit | $74$ | $20$ | Mango | Healthy Fruit | $348$ |
| $9$ | Citrus | Canker Leaf | $210$ | $21$ | Papaya | Ring Spot | $717$ |
| $10$ | Citrus | Healthy Leaf | $192$ | $22$ | Papaya | Healthy Leaf | $298$ |
| $11$ | Citrus | Disease Fruit | $158$ | $23$ | Papaya | Disease Fruit | $127$ |
| $12$ | Citrus | Healthy Fruit | $190$ | $24$ | Papaya | Healthy Fruit | $143$ |
| | | | | | **Total** | **$24$ Classes** | **$10,113$** |

</details>

---

## Repository Structure (Template)

```text
PlantfyCropNetLight/
├── assets/                  # Architecture diagrams & sample dataset figures (TCP.png, ABCGMP-55.png)
├── configs/                 # Dataset-specific hyperparameters (ReLU for TCP, Swish for ABCGMP)
├── data/                    # Dataset directory & preprocessing scripts
├── models/                  # PlantfyCropNetLight architecture & CBAM modules
├── train.py                 # Training and validation pipeline
├── evaluate.py              # Model evaluation & ablation study scripts
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/PlantfyCropNetLight.git
cd PlantfyCropNetLight
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train & Evaluate
```bash
# Train on TCP dataset (uses ReLU in CBAM MLP)
python train.py --dataset TCP --activation relu

# Train on ABCGMP dataset (uses Swish in CBAM MLP)
python train.py --dataset ABCGMP --activation swish
```

---

## Citation

If you find **PlantfyCropNetLight** or the benchmark datasets useful in your research, please cite our work as well as the respective dataset papers:

```bibtex
@article{plantfycropnetlight2026,
  title={PlantfyCropNetLight: A Lightweight Multi-Branch Attention Network for Fine-Grained Multi-Crop Pathology Classification},
  author={Your Name and Co-authors},
  journal={Journal / Conference Name},
  year={2026}
}
```

### Dataset References
* **TCP Dataset:** Jain et al., *Novel TCP Crop Disease Dataset*, 2026 (`\cite{jain2026novel}`).
* **ABCGMP Dataset:** Nain et al., *Comprehensive ABCGMP Crop Disease Dataset*, 2026 (`\cite{nain2026comprehensive}`).

---

## Future Work
* Cross-field domain adaptation for unseen agricultural environments.
* Quantization and pruning for real-time mobile and microcontroller (on-device) deployment.
