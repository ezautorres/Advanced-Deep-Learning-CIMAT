# Advanced Deep Learning – CIMAT (Fall 2024)

**Author:** Ezau Faridh Torres Torres  
**Advisor:** Dr. Mariano Rivera Meraz 
**Course:** Advanced Deep Learning  
**Institution:** CIMAT – Centro de Investigación en Matemáticas  
**Term:** Fall 2024 

*Description*

## 📄 Table of Contents

- [Repository Structure](#repository-structure)
- [Technical Stack](#technical-stack)
- [Overview of Assignments](#overview-of-assignments)
  - [Assignment 1 – LU and Cholesky Decomposition](#assignment-1--lu-and-cholesky-decomposition)
  - [Assignment 2 – QR Decomposition and Least Squares](#assignment-2--qr-decomposition-and-least-squares)
  - [Assignment 3 – Numerical Stability](#assignment-3--numerical-stability)
  - [Assignment 4 – Eigenvalue Computation](#assignment-4--eigenvalue-computation)
  - [Assignment 5 – Stochastic Simulation](#assignment-5--stochastic-simulation)
  - [Final Project – Bayesian Inference for Weibull Parameters](#final-project--bayesian-inference-for-weibull-parameters)
- [Contact](#-contact)

---B

## Repository Structure

Each assignment comprises the following elements:

- Python scripts with modular implementations of the required models and methods.
- A `report.pdf` that explains the methodology and findings.
- A `results/` directory with visual representations of the results.  

---

## Technical Stack

This project was developed in Python 3.11 using:

- **Core libraries:** `numpy`, `scipy`, `matplotlib`, `pandas`
- **Symbolic computation:** `sympy`
- **Statistical modeling & distributions:** `scipy.stats`
- **Plotting & visualization:** `seaborn`, `matplotlib`
- **Jupyter Notebooks** (for prototyping)

> Note: Each assignment may include additional libraries specified in the corresponding script headers.

---

## Overview of Assignments

The following section presents a concise overview of each task, highlighting its primary objective:

### Assignment 1 – Extreme Learning Machines 
Implementation and comparison of a multilayer perceptron (`MLP`), a standard extreme learning machine (`ELM`), and a binary-weight ELM for emotion classification using `VQ-VAE` encoded inputs. The study evaluates the impact of different regularization strategies (none, Ridge, Lasso, ElasticNet) on the ELM’s output layer, using 12×12 integer matrices as input representations of facial expressions.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment1/output.png" alt=" Assignment 1" width="500"/>
</div>

### Assignment 2 – Seq2Seq Prediction for Cryptocurrency Time Series  
Implementation of a sequence-to-sequence (`Seq2Seq`) model with attention and teacher forcing to predict future values in multivariate time series of cryptocurrency prices. Using data from 7 cryptocurrencies (including `Bitcoin`) over 100 hourly intervals, the model forecasts the final segment of each series. Historical data is fetched via yahoo-finance and normalized using MinMax scaling. The model is trained for 300 epochs with LSTM layers of 1000 hidden units.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment2/output.png" alt=" Assignment 2" width="500"/>
</div>

### Assignment 3 – Transformer Encoder for Time Series Forecasting  
Implementation of a Transformer encoder model in `TensorFlow` to predict future cryptocurrency prices from past sequences. The model is built from scratch using `MultiHeadAttention` layers, with three encoding blocks and a latent dimension of 64. Two versions are explored: one using `logarithmic normalization`, and another using `MinMax scaling`. The model’s performance is compared against a naive baseline (no change in price) and the `Seq2Seq` architecture from Assignment 2.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment3/output.png" alt=" Assignment 3" width="500"/>
</div>

### Assignment 4 – Transfer Learning and LoRA for Currency Forecasting 
Extension of the Transformer model from Assignment 3 to a new dataset including exchange rates for multiple currencies and daily oil prices. Three strategies are compared: (1) training a new model from scratch, (2) full fine-tuning of the pretrained transformer, and (3) fine-tuning using Low-Rank Adaptation (`LoRA`) on affine layers. Multiple `LoRA` ranks are tested to evaluate efficiency vs. performance trade-offs.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment4/output.png" alt=" Assignment 4" width="500"/>
</div>

### Assignment 5 – Diffusion Model with Transformer Sampler  
Adaptation of a `DDIM-based diffusion model` by replacing the original `U-Net` architecture with a custom Transformer for the reverse sampling process. The architecture uses 12 attention heads across 8 layers, maintaining all other parameters from the original implementation (e.g., number of diffusion steps, embedding size, learning rate). The model is trained for 50 epochs due to high computational cost.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment5/output.png" alt=" Assignment 5" width="500"/>
</div>

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment5/output2.png" alt=" Assignment 5" width="500"/>
</div>

---

### Final Project – State-Exchange Attention (SEA) for Physics Transformers  
Investigation and replication of the SEA architecture proposed by Esmati et al. (2024), which integrates a novel State-Exchange Attention mechanism into transformer-based models for simulating PDE-governed physical systems. The SEA module enables dynamic cross-field communication between state variables such as velocity, pressure, and volume fraction, effectively reducing autoregressive rollout error. The full ViT-SEA framework achieves up to 91% error reduction compared to state-of-the-art baselines, demonstrating its capacity to capture complex spatiotemporal dynamics in computational fluid dynamics scenarios.

---

## Learning Outcomes

Throughout the course, I gained practical experience in:

- Implementing numerical linear algebra algorithms from scratch
- Performing polynomial and spline interpolation
- Solving ordinary differential equations using numerical schemes
- Designing and evaluating stochastic simulation pipelines (e.g., ARS, MCMC)
- Analyzing convergence and stability in numerical methods
- Applying Bayesian inference via MCMC techniques to real data
- Writing clear scientific reports with integrated visualizations

---

## 📫 Contact

- 📧 Email: ezau.torres@cimat.mx  
- 💼 LinkedIn: [linkedin.com/in/ezautorres](https://linkedin.com/in/ezautorres)