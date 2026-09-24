# Phase 4 – CNN Architecture with Transfer Learning & Grad-CAM

## Executive Overview

Phase 4 establishes the deep learning architecture for classifying Diabetic Retinopathy (DR) from retinal fundus photographs. Building upon the stratified train/val/test splits (Phase 1), image preprocessing (Phase 2), and train-only class-balanced augmentation (Phase 3), this phase implements a **transfer learning pipeline in PyTorch** tailored for training and evaluation in **Google Colab** (with free-tier GPU acceleration) and deployment in the interactive clinical interface.

The architecture is implemented in [`src/model.py`](../../src/model.py) with visual explainability provided by [`src/gradcam.py`](../../src/gradcam.py).

---

## 1. Unified Model Design: Dual Requirement Satisfaction

A central requirement of the assignment brief is:
> *"Use a suitable CNN architecture with transfer learning to classify diabetic retinopathy as well as the stage of the disease."*
> — *Coursework Brief & Assessment Rubric, Criterion 4*

Our architecture accomplishes **both tasks simultaneously** from a single, unified 5-class output:

```
                          [ Retinal Fundus Image (3 x 224 x 224) ]
                                            │
                                            ▼
                           [ Pretrained Backbone (CNN Base) ]
                                            │
                                            ▼
                           [ Custom Classification Head ]
                                            │
                                            ▼
                       [ Raw Logits & Softmax Probabilities ]
                          P = [p0, p1, p2, p3, p4] (sum = 1.0)
                                            │
                   ┌────────────────────────┴────────────────────────┐
                   ▼                                                 ▼
     Task A: Binary DR Presence                        Task B: Severity Staging
    ────────────────────────────                      ──────────────────────────
    P(DR Positive) = p1+p2+p3+p4                      Predicted Grade = argmax(p_i)
                   = 1.0 - p0                         0: No_DR
    Decision: Present if P >= 0.50                    1: Mild
    (Clinician screening filter)                      2: Moderate
                                                      3: Severe
                                                      4: Proliferative_DR
```

### Mathematical Formulation
Given the network output logits $\mathbf{z} = [z_0, z_1, z_2, z_3, z_4]$, the normalized class posterior distribution is:
$$p_i = \frac{e^{z_i}}{\sum_{j=0}^{4} e^{z_j}}, \quad i \in \{0, 1, 2, 3, 4\}$$

1. **Disease Detection (Presence vs. Absence)**:
   $$P(\text{DR Positive}) = \sum_{k=1}^{4} p_k = 1.0 - p_0$$
   $$\hat{y}_{\text{binary}} = \begin{cases} 1 \quad (\text{DR Detected}), & \text{if } P(\text{DR Positive}) \ge \tau \\ 0 \quad (\text{Normal / No DR}), & \text{otherwise} \end{cases}$$
   where default screening threshold $\tau = 0.50$ (tunable to maximize clinical recall).

2. **Disease Staging (Severity Grading)**:
   $$\hat{y}_{\text{stage}} = \arg\max_{i \in \{0, 1, 2, 3, 4\}} p_i$$

### Why a Single Unified Model Is Clinically & Methodologically Superior
* **No Cascading Error**: A two-stage pipeline (binary classifier followed by a 4-class staging classifier) suffers from compounding errors: if the binary stage misclassifies an early Mild case as normal, the staging model never gets to inspect it.
* **Shared Learned Representation**: The features that indicate DR presence (microaneurysms, hemorrhages) are the identical anatomical substrates that determine stage progression. Multi-task sharing reinforces feature learning.
* **Compute & Latency Efficiency**: Only a single forward pass (~15 ms on GPU) is required per patient eye during clinical screening and interactive UI deployment.

---

## 2. Backbone Architecture Selection & Justification

We selected **EfficientNet-B0** as the primary backbone, with full modular support for **ResNet-50** as a comparative baseline.

### Comparative Evaluation

| Feature / Metric | EfficientNet-B0 (Primary) | ResNet-50 (Baseline) | Clinical / Practical Impact |
|:---|:---:|:---:|:---|
| **Total Parameters** | **4,337,281 (~4.3M)** | 24,034,373 (~24.0M) | EfficientNet is **5.5× lighter**, drastically reducing memory overhead |
| **Model Size on Disk** | **~17.4 MB** | ~96.1 MB | Fast loading into web/cloud UI and Colab runtime |
| **Architecture Paradigm** | Compound Scaling + MBConv + Squeeze-and-Excitation (SE) | Identity Residual Blocks ($F(x) + x$) | SE blocks dynamically weight lesion-informative channels |
| **Top-1 ImageNet Accuracy** | **77.1%** | 76.1% | Higher feature representation capacity despite fewer parameters |
| **Colab Free T4 GPU Epoch Time** | **~2.5 – 3.5 minutes** | ~7.0 – 9.5 minutes | Enables rapid hyperparameter tuning within Colab timeout limits |
| **GPU Memory (Batch Size 32)** | **~2.1 GB VRAM** | ~6.8 GB VRAM | Eliminates Out-Of-Memory (OOM) risks on free-tier 15GB T4 GPU |
| **Risk of Overfitting Minority Classes**| **Low** (Compact capacity) | **Moderate to High** (Overparameterized) | Severe (135 train samples) requires tight regularisation |

### Why EfficientNet-B0 Excels for Diabetic Retinopathy
1. **Squeeze-and-Excitation (SE) Channel Attention**: Retinal fundus images consist of extensive uniform reddish backgrounds where critical diagnostic lesions occupy tiny pixel clusters (microaneurysms $\le 25\,\mu\text{m}$, subtle intraretinal microvascular abnormalities). SE attention blocks dynamically recalibrate channel feature maps by explicitly modeling channel interdependencies, selectively amplifying channels that capture lesion textures while dampening homogeneous background channels.
2. **Compound Scaling**: Rather than arbitrarily scaling depth (layers) or width (channels), EfficientNet uniformly scales depth, width, and resolution with a fixed compound coefficient, preserving spatial fidelity without excessive parameter bloat.
3. **Colab Resource Constraints**: The assignment requires systematic experimental evaluation and hyperparameter tuning (Learning rate, weight decay, unfreeze depth). EfficientNet-B0 converges in under 20 epochs (~45 minutes total on T4), allowing multiple experiments within Colab's standard session disconnect limits.

---

## 3. Transfer Learning Strategy & Freezing Scheme

Transfer learning is implemented through a disciplined, two-stage curriculum:

```
[ STAGE 1: Feature Extraction / Head Warmup ]
┌──────────────────────────────────────────────┬──────────────────────────────────┐
│ Backbone: EfficientNet-B0 (FROZEN: 4,007,548)│ Head (TRAINABLE: 329,733, 7.6%)  │
│ ImageNet Pretrained Weights Preserved        │ Random Weights Initialized       │
└──────────────────────────────────────────────┴──────────────────────────────────┘
   • Prevents large gradient backpropagation from destroying pretrained representations
   • Trains custom projection & classification head for 3–5 epochs (lr = 1e-3)

                                      │
                                      ▼
[ STAGE 2: Differential Fine-Tuning ]
┌─────────────────────────┬───────────────────────────────┬───────────────────────┐
│ Early Backbone (FROZEN) │ Top 2 MBConv Blocks (UNFROZEN)│ Head (TRAINABLE)      │
│ Low-level features      │ High-level lesion semantics   │ Task-specific mapping │
│ (2,878,156 params, 66%) │ (1,129,392 params, 26%)       │ (329,733 params, 8%)  │
└─────────────────────────┴───────────────────────────────┴───────────────────────┘
   • Unfreezes top stages (Stages 7 & 8 in EfficientNet-B0)
   • Uses reduced learning rate (lr = 1e-4 or 1e-5) to fine-tune retinal abstractions
```

### Layer Freezing Rationale
* **Early Convolutional Layers (Stem to Stage 6)**: Learn low-level Gabor-like filters (edges, color gradients, texture primitives). These primitives are universal across natural images and medical fundus photographs. Retraining them on only 6,408 images risks feature corruption and catastrophic forgetting.
* **Top Convolutional Blocks (Stages 7 & 8)**: Contain receptive fields broad enough to synthesize high-level spatial relationships (e.g. neovascular fronds originating from the optic disc, macular exudate clusters). Unfreezing these blocks allows the model to specialize its high-level semantics from ImageNet objects to retinal anatomy.

---

## 4. Custom Classification Head Architecture

The classification head is engineered specifically to prevent overfitting while preserving subtle discriminative signals:

```python
nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),      # Converts (1280, 7, 7) feature map -> (1280, 1, 1) vector
    nn.Flatten(),                      # Vector dimension: 1280
    nn.Dropout(p=0.4),                 # Heavy dropout regularisation before dense layer
    nn.Linear(1280, 256),              # Intermediate bottleneck projection
    nn.BatchNorm1d(256),               # Normalizes latent activations; stabilizes gradient flow
    nn.SiLU(inplace=True),             # Smooth, non-monotonic Swish activation
    nn.Dropout(p=0.2),                 # Secondary dropout regularisation
    nn.Linear(256, 5)                  # Final linear projection to 5 raw logits
)
```

### Design Rationales:
1. **Adaptive Average Pooling**: Replaces older fully-connected flatten layers (like VGG's 4096-wide layers) that accounted for 80% of parameters. It makes the model resolution-independent and drastically reduces parameter count.
2. **Intermediate Projection (1280 $\to$ 256)**: Stepping down gradually from 1280 features to 5 classes avoids information bottlenecks and provides an embedding space that can be inspected for feature clustering (t-SNE).
3. **Dual Dropout (p=0.4, p=0.2)**: Essential to counter overfitting on minority classes (Severe and Proliferative DR) where visual variance is high but unique patient instances are limited.
4. **Batch Normalization (1D)**: Accelerates convergence, mitigates internal covariate shift during fine-tuning, and reduces sensitivity to learning rate oscillations.
5. **Raw Logits Output**: The final layer outputs unnormalized logits rather than applying softmax internally. This enables the use of `torch.nn.CrossEntropyLoss` with log-sum-exp numerical stabilization, preventing precision underflow when probabilities approach 0.0.

---

## 5. Model Summary & Parameter Audit

Below is the verified diagnostic summary generated by `src/model.py`:

```
================================================================================
  TRANSFER LEARNING MODEL ARCHITECTURE: EFFICIENTNET_B0
  Diabetic Retinopathy Stage Detection & Binary Classification
================================================================================

[1] SPECIFICATIONS & HYPERPARAMETERS:
  - Backbone Architecture    : efficientnet_b0
  - Pretrained Weights       : ImageNet-1K (Default)
  - Input Dimensions         : (1, 3, 224, 224)
  - Output Target Classes    : 5 (No_DR, Mild, Moderate, Severe, Proliferative_DR)
  - Primary Dropout Rate     : 0.4
  - Projection Hidden Dim    : 256
  - Grad-CAM Target Layer    : Conv2dNormActivation (Stage 8, 1280 channels)

[2] STAGE 1: FEATURE EXTRACTION (Base Layers Frozen)
  - Total Parameters         : 4,337,281
  - Trainable (Head only)    : 329,733 (7.60%)
  - Frozen (Backbone)        : 4,007,548 (92.40%)

[3] STAGE 2: FINE-TUNING (Top 2 Blocks + Head Unfrozen)
  - Total Parameters         : 4,337,281
  - Trainable Parameters     : 1,459,125 (33.64%)
  - Frozen Parameters        : 2,878,156 (66.36%)

[4] FULL UNFREEZE REFERENCE (All Layers Trainable)
  - Total Parameters         : 4,337,281
  - Trainable Parameters     : 4,337,281 (100.00%)
  - Frozen Parameters        : 0 (0.00%)

[5] CLASSIFICATION HEAD LAYERS:
  [0] AdaptiveAvgPool2d      Parameters:          0
  [1] Flatten                Parameters:          0
  [2] Dropout (p=0.4)        Parameters:          0
  [3] Linear (1280 -> 256)   Parameters:    327,936
  [4] BatchNorm1d (256)      Parameters:        512
  [5] SiLU                   Parameters:          0
  [6] Dropout (p=0.2)        Parameters:          0
  [7] Linear (256 -> 5)      Parameters:      1,285
  --------------------------------------------------
  Total Head Parameters     :                329,733

[6] FORWARD PASS & DUAL OBJECTIVE VERIFICATION:
  - Input Tensor Shape       : [1, 3, 224, 224]
  - Output Logits Shape      : [1, 5]
  - Softmax Probabilities    : Generated and normalized (sum = 1.0)
  - Dual Output Derivation   : Verified (Stage argmax & DR positive prob 1 - p0)
  - Grad-CAM Verification    : PASSED (Shape: 224 x 224, Range: [0.00, 1.00])
================================================================================
```

---

## 6. Grad-CAM Explainability Utility

In medical computer vision, high accuracy alone is insufficient: models must be **clinically trustworthy**. Deep learning models are vulnerable to learning "shortcut features" (e.g. camera manufacturer markings, black circular border artifacts, or lens dust spots).

[`src/gradcam.py`](../../src/gradcam.py) implements **Gradient-Weighted Class Activation Mapping (Grad-CAM)**:

### How It Works:
1. Attaches forward and backward PyTorch hooks to the last convolutional layer (`features[-1]` in EfficientNet-B0).
2. During the backward pass for a target class $c$, captures gradients $\frac{\partial y^c}{\partial A^k}$ representing the sensitivity of the class score to feature map $k$.
3. Computes neuron importance weights via Global Average Pooling:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
4. Computes the weighted combination and applies ReLU to isolate features with a positive evidentiary contribution:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
5. Normalizes the map to $[0, 1]$ and upsamples to $224 \times 224$, overlaying it as a Jet colormap on the fundus photograph.

### Clinical Use in Phase 5:
* **Microaneurysm Localization**: Proves the model fires on tiny red dots in Grade 1 (Mild).
* **Macular Exudate Verification**: Confirms intense activation over yellow lipid deposits in Grades 2 & 3.
* **Neovascular Fronds**: Validates detection of irregular new vessel proliferation at the optic disc in Grade 4.

---

## 7. Hyperparameter Tuning Plan for Phase 5 (Colab Training)

To satisfy **Rubric Criterion 4 (CNN Architecture & Transfer Learning, 20 Marks)** and **Criterion 5 (Training Strategy & Experimental Design, 10 Marks)**, systematic hyperparameter experiments will be executed in Phase 5.

Below are the specific hyperparameters planned for tuning and their clinical rationales:

| Hyperparameter | Planned Search Grid | Primary Baseline | Why It Matters for Diabetic Retinopathy |
|:---|:---:|:---:|:---|
| **Learning Rate (Stage 1: Head)** | `[5e-4, 1e-3, 3e-3]` | `1e-3` | Head has random weights; needs high enough rate to adapt rapidly without oscillating. |
| **Learning Rate (Stage 2: Backbone)** | `[1e-5, 5e-5, 1e-4]` | `5e-5` | Must be 10–20× smaller than Stage 1 to prevent catastrophic forgetting of pretrained weights. |
| **Optimizer** | `[AdamW, Adam, SGD+Momentum]` | `AdamW` | `AdamW` decouples weight decay from gradient updates, providing superior regularisation for medical images. |
| **Weight Decay ($L_2$)** | `[1e-4, 1e-3, 1e-2]` | `1e-3` | Constrains weight magnitudes; critical for minority classes (Severe, Proliferative) to prevent overfitting. |
| **Batch Size** | `[16, 32, 64]` | `32` | 32 balances gradient estimation stability, GPU memory occupancy, and generalization regularisation on Colab T4. |
| **Unfreeze Depth (Stage 2)** | `[Top 1 block, Top 2 blocks, Full]` | `Top 2 blocks` | Determines trade-off between domain specialization (retinal features) vs. overfitting. |
| **Class Loss Weighting** | `[Unweighted, Inverse-Frequency, Effective-Num]` | `Effective-Num` | Even after augmentation, minor residual class imbalances can bias gradients toward dominant classes. |
| **Head Dropout Rate** | `[0.3, 0.4, 0.5]` | `0.4` | Primary control against co-adaptation of features in the dense projection layer. |

---

## 8. Google Colab Execution Guide

Because full training of 6,408 augmented images across 20–25 epochs requires GPU acceleration, Phase 5 will be run on Google Colab. The codebase is designed with **zero hardcoded platform paths**, allowing direct cloning and execution:

```bash
# 1. Clone the repository in Colab:
!git clone -b feature/phase-4-cnn-model https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment.git
%cd diabetic-retinopathy-cv-assignment

# 2. Verify GPU availability:
import torch
print("GPU Available:", torch.cuda.is_available())
print("Device Name  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# 3. Test model instantiation and parameter verification:
!python src/model.py --backbone efficientnet_b0
```

---

*Phase 4 completed by Sanduni Herath. Architecture artifacts and model definitions are fully version-controlled on `feature/phase-4-cnn-model`.*
