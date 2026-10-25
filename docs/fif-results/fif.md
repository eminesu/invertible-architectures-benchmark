# FIF Training Configuration

Architecture options, shared training settings, and problem-specific parameters

## 1.0 Overview

For FIF, we used **two architectures: the original MLP and the paper (p3) variant**.

The shared training settings were:

### Shared Training Settings

| Parameter | Value |
|---|---|
| Training / validation samples | 1,000,000 / 20,000 |
| Epochs / batch size | 50 / 1,000 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Learning-rate schedule | Cosine annealing |
| Weight decay | 1e-5 |
| Reconstruction weight β | 100 |
| Gradient clipping | 10 |
| Parameter budget | 3 million |
| Seed / device | 0 / CUDA |
| KDE centers | 1,024 |

## 2.0 Architecture Settings

The architecture settings were:

### Architecture Comparison

| Experiment | Hidden width | Hidden layers per MLP | Total parameters |
|---|---:|---:|---:|
| Kinematics, original MLP | 701 | 4 | 2,996,080 |
| Ballistics, original MLP | 702 | 4 | 2,994,739 |
| Kinematics, paper | 697 | 2 | 2,992,932 |
| Ballistics, paper | 698 | 2 | 2,991,647 |

### Activations and Residual Blocks

The original MLP used **SiLU**. The paper variant used **ReLU**, with **two latent residual blocks each in the encoder and decoder**, and two hidden layers per block.

## 3.0 Problem-Specific Settings

Problem-specific settings:

- **Kinematics:** `x_dim=4`, `y_dim=2`, `z_dim=2`, observation noise `σ=0.01`.
- **Ballistics:** `x_dim=4`, `y_dim=1`, `z_dim=3`, `σ=0.02`. For discrete speed, we added `U(-0.5, 0.5)` noise during training and rounded generated speeds during sampling.

## 4.0 Shared Modeling and Selection Approach

Both variants used a **fixed prior** and **exact trace computation**. We selected checkpoints using **validation re-simulation error**.

## 5.0 Why FIF Is Well Suited to These Problems

FIF is well suited to these inverse problems because multiple parameter combinations can produce the same observation. Its lower-dimensional latent space provides a natural representation of the uncertainty that remains after conditioning on an observation.

### 1. Geometry Aligned with the Inverse Problem

Kinematics has four unknown parameters and two observed coordinates. Fixing the observation generally leaves two degrees of freedom, consistent with **z_dim=2**. Ballistics has four parameters and one observation, motivating **z_dim=3** under our continuous approximation. This lower-dimensional solution structure is precisely what injective flows are designed to model.

### 2. Diverse Plausible Solutions

For a fixed observation, varying the latent sample produces different parameter combinations. FIF combines a reconstruction term with a density-learning objective, aiming to learn both the solution set and the probability distribution over that set. Whether all posterior modes are recovered must still be verified empirically.

### 3. Flexible Network Design

FIF permits separate encoder and decoder networks without requiring coupling layers. This flexibility allows standard MLPs or residual blocks to approximate the curved relationships induced by our simulators.

### 4. Efficient Sampling

Our implementation generates parameters directly through the decoder, without numerical ODE integration. For the original MLP, generating one million samples took approximately **3.04 seconds** for kinematics and **3.27 seconds** for ballistics on CUDA, counting sampling time only.

**Implementation qualifications.** Noisy observations require a distribution over clean outputs, which we approximate using a KDE bridge. Ballistics also includes a discrete speed variable, handled through dequantization during training and rounding during sampling. Consequently, the geometric motivation is strongest for kinematics, whereas ballistics requires additional care.

## 6.0 FIF Architectures Used in Our Experiments

In the tables, **C stands for condition**:

- **Kinematics:** The desired arm-tip position: horizontal and vertical coordinates, $(u^*, v^*)$.
- **Ballistics:** The desired horizontal landing position, $x_{\mathrm{impact}}^*$.

The condition is applied through scale and shift before each hidden-layer activation. Its dimension is 2 for kinematics and 1 for ballistics.

### Original MLP Architecture

| Component | Kinematics | Ballistics |
|---|:---:|:---:|
| **Encoder input** | **(4,)** | **(4,)** |
| Layer 1 | Linear(4, 701), C, SiLU | Linear(4, 702), C, SiLU |
| Layers 2–4 | 3 × [Linear(701, 701), C, SiLU] | 3 × [Linear(702, 702), C, SiLU] |
| Layer 5 | Linear(701, 2) | Linear(702, 3) |
| **Decoder input** | **(2,)** | **(3,)** |
| Layer 1 | Linear(2, 701), C, SiLU | Linear(3, 702), C, SiLU |
| Layers 2–4 | 3 × [Linear(701, 701), C, SiLU] | 3 × [Linear(702, 702), C, SiLU] |
| Layer 5 | Linear(701, 4) | Linear(702, 4) |
| **Normalization** | **None** | **None** |
| **#Parameters** | **2,996,080** | **2,994,739** |

### Paper-Inspired Architecture (with Resblocks)

[Appendix E.2 of the FIF paper](https://arxiv.org/html/2306.01843v5#A5.SS2)

| Component | Kinematics | Ballistics |
|---|:---:|:---:|
| **Encoder input** | **(4,)** | **(4,)** |
| Layer 1 | Linear(4, 697), C, ReLU | Linear(4, 698), C, ReLU |
| Layer 2 | Linear(697, 697), C, ReLU | Linear(698, 698), C, ReLU |
| Layer 3 | Linear(697, 2) | Linear(698, 3) |
| Layers 4–5 | 2 × ResBlock(2, 697) | 2 × ResBlock(3, 698) |
| **Decoder input** | **(2,)** | **(3,)** |
| Layers 1–2 | 2 × ResBlock(2, 697) | 2 × ResBlock(3, 698) |
| Layer 3 | Linear(2, 697), C, ReLU | Linear(3, 698), C, ReLU |
| Layer 4 | Linear(697, 697), C, ReLU | Linear(698, 698), C, ReLU |
| Layer 5 | Linear(697, 4) | Linear(698, 4) |
| **Normalization** | **None** | **None** |
| **#Parameters** | **2,992,932** | **2,991,647** |
