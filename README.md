# Federated Learning-Based Brain Tumor Segmentation
### Using 3D U-Net and DeepLabV3+ Architectures

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange?style=flat-square&logo=pytorch)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square&logo=google-colab)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4-green?style=flat-square&logo=nvidia)
![Conference](https://img.shields.io/badge/IEEE-ICBDML%202026-red?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-BraTS--GLI%202024-purple?style=flat-square)

**Accepted and Presented at IEEE ICBDML 2026 — Paper ID: ICBDML-2026-738**

*Pronomi Roy · Adarsh S Nair · Guide: Kalpana C*
*Department of Computational Intelligence, SRMIST Kattankulathur*

</div>

---

## Table of Contents

- [Overview](#overview)
- [What Makes This Different](#what-makes-this-different)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Federated Protocol](#federated-protocol)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Overview

Brain tumor segmentation from multi-modal MRI is a clinically critical task that directly guides surgical planning, radiation targeting, and treatment monitoring for glioma patients. While deep learning has achieved impressive segmentation accuracy, hospitals cannot legally share raw patient MRI data across institutions under HIPAA and GDPR — making it impossible to train large centralised models on diverse multi-institutional data.

This project solves that problem through a **heterogeneous federated learning framework** where:

- **Hospital A** runs a **3D U-Net** — processes full volumetric MRI scans, captures cross-slice spatial context
- **Hospital B** runs a **DeepLabV3+** — processes 2D axial slices, captures boundary-aware multi-scale features
- A **Meta-Fusion Server** combines compact learned representations from both hospitals and sends back a guidance signal that improves both models — without any raw patient data ever leaving either institution

The system operates across **three communication rounds**, each round producing a richer guidance signal that progressively improves segmentation accuracy on both clients.

---

## What Makes This Different

Every existing federated medical imaging system requires all hospitals to run **identical model architectures**. This is unrealistic — different hospitals invest in different hardware and systems.

| Feature | Existing FL Systems | This Project |
|---------|-------------------|--------------|
| Architecture requirement | All clients identical | **Heterogeneous — 3D + 2D** |
| Data shared | Gradients / weights | **Nothing — features only** |
| Privacy level | Weight sharing (vulnerable) | **Feature-level (stronger)** |
| Hardware requirement | High-end uniform | **Commodity T4 GPU** |
| Cross-architecture fusion | Not possible | **Slice-stacking alignment** |

The key technical contribution is the **slice-stacking feature alignment strategy** that produces identically-shaped tensors `(B, 256, 16, 16, 16)` from both the 3D U-Net bottleneck and the DeepLabV3+ ASPP output, enabling cross-institutional fusion despite completely different architectures.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEDERATED SYSTEM OVERVIEW                     │
└─────────────────────────────────────────────────────────────────┘

  HOSPITAL A                                        HOSPITAL B
  ──────────                                        ──────────
  Raw MRI (stays local)                             Raw MRI (stays local)
       │                                                  │
  ┌────▼────────────────┐                    ┌────────────▼────────┐
  │    3D U-Net          │                    │   DeepLabV3+         │
  │  Input: (B,4,128³)   │                    │  Input: (B,4,128,128)│
  │  Encoder × 3 stages  │                    │  ResNet-50 Backbone  │
  │  Bottleneck: 16³     │                    │  ASPP: rates 6,12,18 │
  │  Output: (B,4,128³)  │                    │  Output: (B,4,128²)  │
  └────────────────┬─────┘                    └──────┬───────────────┘
                   │                                  │
        feat_3d (20,256,16³)              feat_2d (20,256,16³)
        [slice-stacking alignment]        [ASPP × 16 depth slices]
                   │                                  │
                   └──────────────┬───────────────────┘
                                  │
                      ┌───────────▼───────────┐
                      │   META-FUSION SERVER   │
                      │   ~4.8M parameters     │
                      │                        │
                      │  align_3d → 256ch      │
                      │  align_2d → 256ch      │
                      │  cat → 512ch           │
                      │  fusion1 → 256ch       │
                      │  fusion2 → 128ch       │
                      │  fusion3 →  64ch       │
                      │  ┌──────────────────┐  │
                      │  │  seg_head        │  │
                      │  │  (training only) │  │
                      │  └──────────────────┘  │
                      │  ┌──────────────────┐  │
                      │  │  guidance_head   │  │
                      │  │ (20,256,16,16,16)│  │
                      │  └────────┬─────────┘  │
                      └───────────┼────────────┘
                                  │
              guidance_roundN.pt  │
                   ┌──────────────┴───────────────┐
                   │                              │
        ┌──────────▼──────────┐       ┌───────────▼────────────┐
        │  3D U-Net Round N   │       │  DeepLabV3+ Round N    │
        │  Inject at          │       │  Inject at ASPP output │
        │  bottleneck         │       │  guidance.mean(dim=2)  │
        │  cat(btl, guid)     │       │  cat(aspp, guid_2d)    │
        │  Conv3D(512→256,1)  │       │  Conv2D(512→256,1)     │
        └─────────────────────┘       └────────────────────────┘
```

---

## Dataset

**BraTS-GLI 2024 Challenge Dataset**

| Property | Details |
|----------|---------|
| Task | Glioma segmentation on pre-operative MRI |
| Modalities | T1n, T1ce, T2w, T2-FLAIR (4 per patient) |
| Volume size | 240 × 240 × 155 voxels (cropped to 128³) |
| Tumor classes | Background (0), NCR (1), Edema (2), ET (3) |
| Client 1 split | 375 train / 125 val / 40 test |
| Client 2 split | 375 train / 125 val / 40 test |
| Total cases used | 500 per client (1000 total) |
| Test patients | 80 completely held-out (40 per client) |

Download: [BraTS 2024 Challenge](https://www.synapse.org/brats2024)

---

## Project Structure

```
BraTS2024-Federated-Learning/
│
├── notebooks/
│   ├── BraTS2024_3DUNet_clean.ipynb          # Client 1 — Hospital A
│   ├── BraTS2024_Client2_DeepLabV3Plus.ipynb # Client 2 — Hospital B
│   └── BraTS2024_MetaFusion_Server.ipynb     # Server notebook
│
├── models/
│   ├── unet3d.py              # 3D U-Net architecture
│   ├── deeplabv3plus.py       # DeepLabV3+ with ASPP
│   └── meta_fusion_server.py  # Meta-Fusion Server
│
├── utils/
│   ├── preprocessing.py       # NIfTI loading, normalization, cropping
│   ├── losses.py              # Dice + Focal + BCE combined loss
│   ├── metrics.py             # Dice, IoU, Precision, Recall, HD95
│   └── checkpoint.py          # Save / load / auto-resume
│
├── evaluation/
│   ├── evaluate_unet.py       # 40-patient evaluation — Client 1
│   └── evaluate_deeplabv3.py  # 40-patient evaluation — Client 2
│
├── results/
│   ├── unet_evaluation_summary.txt
│   └── deeplab_evaluation_summary.txt
│
└── README.md
```

---

## Installation

### Requirements

```bash
# Core deep learning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Medical imaging
pip install nibabel

# Segmentation models
pip install segmentation-models-pytorch
pip install segmentation-models-pytorch-3d

# Evaluation
pip install surface-distance

# Utilities
pip install tqdm torchmetrics tensorboard scikit-learn
```

### Google Drive Structure

Set up the following folder structure in your Google Drive before running:

```
MyDrive/
├── Client_1_Hospital_A/          ← BraTS-GLI 2024 data (Account 1)
│   └── BraTS-GLI-XXXXX-XXX/
│       ├── *t1n.nii.gz
│       ├── *t1c.nii.gz
│       ├── *t2f.nii.gz
│       ├── *t2w.nii.gz
│       └── *seg.nii.gz
│
├── client B dataset/
│   └── Client_2_Hospital_B/      ← BraTS-GLI 2024 data (Account 2)
│
├── BraTS2024_3DUNet/             ← Client 1 outputs (Account 1)
│   ├── checkpoints/
│   ├── features/
│   └── preprocessed/
│
├── BraTS2024_DeepLabV3/          ← Client 2 outputs (Account 2)
│   ├── checkpoints/
│   ├── features/
│   └── preprocessed/
│
└── BraTS2024_Server/             ← Server outputs (Account 1)
    ├── server_round1.tar
    ├── server_round2.tar
    ├── guidance_round1.pt
    └── guidance_round2.pt
```

---

## How to Run

The system requires **two Google accounts** to simulate two hospitals and **one server notebook** on Account 1.

### Phase 1 — Independent Pretraining (Run simultaneously)

**Account 1 — Client 1 (3D U-Net)**
```
Open: BraTS2024_3DUNet_clean.ipynb
Run:  Cells 1 → 18 (full pipeline)
Time: ~20 hours on T4 (50 epochs × ~26 min/epoch)
Output: best_ckpt.tar, training curves, inference visualisation
```

**Account 2 — Client 2 (DeepLabV3+)**
```
Open: BraTS2024_Client2_DeepLabV3Plus.ipynb
Run:  Cells 1 → 18 (full pipeline)
Time: ~22 hours on T4 (50 epochs × ~26 min/epoch)
Output: best_ckpt.tar, training curves, inference visualisation
```

### Phase 2 — Feature Extraction (After Phase 1)

**Account 1 — Extract 3D U-Net features**
```python
# Run feature extraction cell in 3D U-Net notebook
# Output: feat_3d_round1.pt  shape (20, 256, 16, 16, 16)  ~84 MB
```

**Account 2 — Extract DeepLabV3+ features**
```python
# Run Cell 19 with ROUND_NUM = 1
# Output: feat_2d_round1.pt  shape (20, 256, 16, 16, 16)  ~84 MB
```

### Phase 3 — Server Training (Account 1)

```
Open:   BraTS2024_MetaFusion_Server.ipynb
Inputs: feat_3d_round1.pt + feat_2d_round1.pt + masks
Run:    Cells 1 → 10
Time:   ~15 minutes on T4 (100 epochs)
Output: server_round1.tar, guidance_round1.pt
```

### Phase 4 — Guided Fine-Tuning Round 2 (Run simultaneously)

```
Account 1: Load guidance_round1.pt → fine-tune 3D U-Net 10 epochs
Account 2: Load guidance_round1.pt → fine-tune DeepLabV3+ 10 epochs
Time: ~4-5 hours each
```

### Phase 5 — Server Update Round 2

```
Inputs: feat_3d_round2.pt + feat_2d_round2.pt
Output: guidance_round2.pt
Time:   ~15 minutes
```

### Phase 6 — Guided Fine-Tuning Round 3 (Final)

```
Account 1: Load guidance_round2.pt → fine-tune 3D U-Net 10 epochs
Account 2: Load guidance_round2.pt → fine-tune DeepLabV3+ 10 epochs
```

### Phase 7 — Final Evaluation

```python
# Run evaluation cells on 40 held-out test patients per client
# Reports: Dice, IoU, Precision, Recall, HD95
# Regions: WT, TC, ET, NCR, Edema
```

---

## Federated Protocol

```
Round   Client 1 (3D U-Net)         Server              Client 2 (DeepLabV3+)
──────  ──────────────────────       ──────              ────────────────────────
  0     Train 50 epochs         →   (waiting)    ←      Train 50 epochs
        Extract feat_3d_r1      →   Fuse features ←      Extract feat_2d_r1
                                    Train 100 ep
                                    Save guidance_r1
  2     Load guidance_r1        ←   Send guidance  →     Load guidance_r1
        Fine-tune 10 epochs          (waiting)           Fine-tune 10 epochs
        Extract feat_3d_r2      →   Fuse features ←      Extract feat_2d_r2
                                    Update server
                                    Save guidance_r2
  3     Load guidance_r2        ←   Send guidance  →     Load guidance_r2
        Fine-tune 10 epochs                              Fine-tune 10 epochs
        Evaluate 40 patients                             Evaluate 40 patients
```

**Data that crosses institutional boundaries:**
- `feat_3d_roundN.pt` — abstract bottleneck features (NOT raw MRI)
- `feat_2d_roundN.pt` — abstract ASPP features (NOT raw MRI)
- `guidance_roundN.pt` — server guidance signal

**Data that NEVER leaves the institution:**
- Raw NIfTI MRI files
- Preprocessed .npy arrays
- Patient metadata
- Model weights

---

## Results

### Client 1 — 3D U-Net (40 Held-Out Test Patients)

| Region | Metric | Before Guidance | After Guidance | Improvement |
|--------|--------|----------------|----------------|-------------|
| WT | Dice | 0.7735 | 0.7822 | +0.0087 |
| WT | IoU | 0.6933 | 0.7010 | +0.0078 |
| WT | HD95 (mm) | 29.46 | 20.69 | ↓ 8.77 |
| NCR | Dice | 0.6685 | 0.7451 | **+0.0766** |
| NCR | HD95 (mm) | 68.09 | 40.02 | ↓ 28.07 |
| Edema | Dice | 0.7412 | 0.7540 | +0.0129 |

### Client 2 — DeepLabV3+ (40 Held-Out Test Patients)

| Region | Metric | Before Guidance | After Guidance | Improvement |
|--------|--------|----------------|----------------|-------------|
| WT | Dice | 0.9192 | 0.9386 | **+0.0193** |
| WT | IoU | 0.8617 | 0.8924 | +0.0307 |
| WT | HD95 (mm) | 2.875 | 2.382 | ↓ 0.493 |
| TC | Dice | 0.8414 | 0.8712 | +0.0297 |
| ET | Dice | 0.8407 | 0.8706 | +0.0299 |

### Meta-Fusion Server Convergence

| Round | Best Loss | Guidance Mean | Guidance Max |
|-------|-----------|---------------|--------------|
| Round 1 | 0.1949 | 0.1783 | 14.7007 |
| Round 2 | 0.1777 | 0.1870 | 14.8999 |

---

## Technologies Used

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | PyTorch 2.10 |
| 3D Segmentation | segmentation-models-pytorch-3d |
| 2D Segmentation | segmentation-models-pytorch |
| MRI Loading | nibabel |
| HD95 Metric | surface-distance |
| Mixed Precision | torch.amp (AMP) |
| Backbone | ResNet-50 (ImageNet pretrained) |
| Hardware | NVIDIA T4 GPU (15.6 GB VRAM) |
| Storage | Google Drive |
| Compute | Google Colab |
| Tracking | TensorBoard |

---

## Key Design Decisions

**Why feature-level federation instead of weight averaging?**
Weight averaging (FedAvg) requires identical architectures. Feature tensors are architecture-agnostic — any model that produces a (B, 256, 16, 16, 16) tensor can participate regardless of how it was built internally.

**Why slice-stacking for DeepLabV3+?**
Direct depth pooling destroys tumor progression information across axial slices. Slice-stacking preserves the full depth structure by processing 16 evenly-spaced slices independently and stacking their ASPP outputs along a new depth axis — no information loss.

**Why gradient accumulation?**
A single 128³ MRI volume with 4 channels requires ~13 GB VRAM during training. The T4 has 15.6 GB total, allowing batch size 1 only. Accumulating gradients over 8 steps mathematically simulates batch size 8 without exceeding VRAM.

**Why Dice + Focal loss?**
Background occupies 85-90% of voxels. Standard cross-entropy learns to predict all-background. Dice loss directly optimises the metric reported in evaluation. Focal loss forces attention on the hard boundary voxels that Dice loss would otherwise ignore.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{roy2026federated,
  title     = {Federated Learning-Based Brain Tumor Segmentation
               Using 3D U-Net and DeepLabV3+ Architectures},
  author    = {Roy, Pronomi and Nair, Adarsh S},
  booktitle = {2nd International IEEE Conference on Big Data
               and Machine Learning (ICBDML 2026)},
  year      = {2026},
  note      = {Paper ID: ICBDML-2026-738},
  address   = {IES College of Technology, Bhopal, India}
}
```

---

## Acknowledgements

- Guide: **Kalpana C**, Assistant Professor, Department of Computational Intelligence, SRMIST
- Dataset: **BraTS-GLI 2024 Challenge** — Adewole et al., arXiv:2405.18368
- Compute: **Google Colab** free tier T4 GPU
- Conference: **IEEE ICBDML 2026**, IES College of Technology, Bhopal

---

<div align="center">

**Department of Computational Intelligence**
**SRM Institute of Science and Technology, Kattankulathur — 603 203**
**May 2026**

</div>
