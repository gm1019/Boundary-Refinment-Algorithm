# Boundary Refinement Network for Liver Tumour Segmentation

## Overview

This repository contains the implementation of the **Boundary Refinement Network (BRN)** proposed in my PhD research for liver tumour segmentation in abdominal CT images.

Instead of directly predicting the entire tumour mask, the proposed framework refines the coarse tumour segmentation by learning only the boundary regions. A patch-based refinement strategy is adopted to improve segmentation accuracy around complex tumour boundaries while preserving the overall tumour structure.

The method is designed as a post-processing refinement stage and can be integrated with different liver tumour segmentation networks.

---

## Highlights

- Patch-based boundary refinement
- Dedicated refinement network for tumour boundaries
- Boundary mask guidance
- Majority voting for overlapping patches
- Improved boundary accuracy with limited computational cost

---

## Framework

The complete framework consists of four stages:

1. Coarse tumour segmentation
2. Boundary extraction
3. Patch refinement network
4. Patch reconstruction and majority voting



---

## Boundary Refinement Strategy

The proposed method only refines pixels located around tumour boundaries.

The coarse segmentation is first converted into a boundary map. Image patches centred on boundary pixels are extracted together with their corresponding coarse segmentation masks. Both inputs are fed into the Boundary Refinement Network, which predicts refined tumour boundaries.

After all boundary patches have been processed, the refined patches are stitched back into the original image using an overlapping voting strategy to generate the final tumour segmentation.

---

## Repository Structure

```text
Boundary-Refinement-Network/
│
├── README.md
├── train.py
├── predict.py
├── visualize.py
├── requirements.txt
│
├── models/
│   └── boundary_refinement_network.py
│
├── dataset/
│   ├── data_loader.py
│   ├── patch_generator.py
│   └── reconstruction.py
│
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── boundary.py
│
├── checkpoints/
│
├── figures/
│
└── results/
```

---

## Training

The Boundary Refinement Network is trained using boundary image patches extracted from coarse tumour segmentation results.

Training images consist of:

- Original CT patch
- Coarse tumour mask patch
- Ground truth boundary patch

---

## Inference

During inference:

1. Generate coarse tumour segmentation.
2. Extract boundary regions.
3. Generate boundary patches.
4. Refine each boundary patch.
5. Reconstruct the full segmentation using majority voting.

---

## Experimental Results

The proposed Boundary Refinement Network improves segmentation performance by refining tumour boundaries while preserving the global tumour shape.

The method demonstrates improvements in:

- Dice Similarity Coefficient (DSC)
- Average Symmetric Surface Distance (ASSD)
- Maximum Surface Distance (MSD)

compared with the coarse segmentation.

---
