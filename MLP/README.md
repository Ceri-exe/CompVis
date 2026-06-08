# MNIST Handwritten Digit Classification using PyTorch

A clean, modular implementation of a baseline Multi-Layer Perceptron (MLP) trained on the MNIST handwritten digit dataset. 

This project served as a study on architectural bottlenecks and evaluation loops using PyTorch.

## Project Structure
* `model.py`: Defines the neural network architecture (`MLP` custom module).
* `train.py`: Handles data engineering pipelines, device mapping, training/validation steps, and loss diagnostics.

## Model Summary & Diagnostics
The architecture uses a traditional single hidden-layer setup:
* **Input Layer:** $28 \times 28 = 784$ units (flattened image arrays)
* **Hidden Layer:** $15$ units with ReLU activation
* **Output Layer:** $10$ units (representing digits 0–9)

### Technical Analysis of Performance
While the training loop runs successfully across CPU, CUDA, and Apple Silicon runtimes, the network encounters an intentional **informational bottleneck**. Compressing $784$ unique features down to just $15$ dimensions restricts the network's capacity to generalize high-variance structural curves inherent to handwritten numerals. 

The evaluation script includes a mismatch analysis utility that visualizes samples where predictions fall short, pointing to an analytical direction for future optimization iterations.

## Future Performance Optimization Ideas
To scale accuracy paste this baseline threshold, subsequent phases of this workspace will implement:
1. **Widening Hyperparameters:** Expanding the hidden layer structure to 128 or 256 units to decrease structural compression.
2. **Optimizer Upgrades:** Moving from traditional Stochastic Gradient Descent (`SGD`) to Adaptive Moment Estimation (`Adam`) for quicker convergence.
3. **Architectural Upgrades:** Implementing a CNN to preserve spatial features instead of flattening input dimensions.