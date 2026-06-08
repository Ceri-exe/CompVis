# MNIST Handwritten Digit Classification using a 2D CNN

A structured PyTorch implementation of a 2D CNN optimized for classifying handwritten digits via the MNIST dataset.

## Project Architecture
* `model.py`: Outlines the 2-stage feature extraction framework (Conv2D -> ReLU -> MaxPool2D) and classification layer.
* `train.py`: Manages data ingestion pipelines, learning loops, filter visualizations, and prediction diagnostic tests.

## Parameter Breakdown
One of the highlights of this setup is its structural efficiency compared to traditional deep dense networks. Below is the analytical audit proving the final sum of **28,938 trainable parameters**:

1. **Convolutional Layer 1 (`conv1.0`)**: 
   * Weights: $16 \text{ filters} \times (1 \text{ input channel} \times 5 \times 5 \text{ kernel size}) = 400$
   * Biases: $16 \text{ biases}$ (one per filter)
   * *Subtotal:* **416 parameters**

2. **Convolutional Layer 2 (`conv2.0`)**:
   * Weights: $32 \text{ filters} \times (16 \text{ input channels} \times 5 \times 5 \text{ kernel size}) = 12,800$
   * Biases: $32 \text{ biases}$ (one per filter)
   * *Subtotal:* **12,832 parameters**

3. **Fully Connected Output Layer (`out`)**:
   * Weights: $10 \text{ classes} \times (32 \text{ channels} \times 7 \times 7 \text{ feature dimensions}) = 10 \times 1,568 = 15,680$
   * Biases: $10 \text{ output biases}$
   * *Subtotal:* **15,690 parameters**

$$\text{Grand Total: } 416 + 12,832 + 15,690 = \mathbf{28,938 \text{ trainable parameters}}$$

## Core Optimizations
* **Adam Optimizer:** Upgraded from classic SGD (See mlp project) to utilize adaptive gradient tracking, securing faster convergence.
* **Spatial Feature Extraction:** Utilized standard padding (`padding=2`) to retain image edge details prior to sub-sampling via downscaling pooling transformations.