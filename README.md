# Advanced Deep Learning – CIMAT (Fall 2024)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-✓-ee4c2c)
![TensorFlow](https://img.shields.io/badge/TensorFlow-✓-ff6f00)
![PINNs](https://img.shields.io/badge/PINNs-✓-pink)
![Neural Networks](https://img.shields.io/badge/Neural%20Networks-✓-hotpink)
![Transformers](https://img.shields.io/badge/Transformers-✓-mediumvioletred)


**Author:** Ezau Faridh Torres Torres  
**Advisor:** Dr. Mariano Rivera Meraz  
**Course:** Advanced Deep Learning  
**Institution:** CIMAT – Centro de Investigación en Matemáticas  
**Term:** Fall 2024 

Comprehensive exploration of modern deep learning architectures—including transformers, recurrent networks, diffusion models, and domain-specific adaptations like LoRA and state-aware mechanisms—applied to time series forecasting and physical system modeling. Each assignment targets a specific modeling challenge and showcases techniques such as attention, transfer learning, and diffusion processes across domains ranging from financial forecasting to PDE-based simulations.

## 📄 Table of Contents

- [Repository Structure](#repository-structure)
- [Technical Stack](#technical-stack)
- [Overview of Assignments](#overview-of-assignments)
  - [Assignment 1 – Extreme Learning Machines](#assignment-1--extreme-learning-machines)
  - [Assignment 2 – Seq2Seq Prediction for Cryptocurrency Time Series](#assignment-2--seq2seq-prediction-for-cryptocurrency-time-series)
  - [Assignment 3 – Transformer Encoder for Time Series Forecasting](#assignment-3--transformer-encoder-for-time-series-forecasting)
  - [Assignment 4 – Transfer Learning and LoRA for Currency Forecasting](#assignment-4--transfer-learning-and-lora-for-currency-forecasting)
  - [Assignment 5 – Diffusion Model with Transformer Sampler](#assignment-5--diffusion-model-with-transformer-sampler)
  - [Final Project – State-Exchange Attention (SEA) for Physics Transformers](#final-project--state-exchange-attention-sea-for-physics-transformers)
- [Contact](#📨-contact)

---

## Repository Structure

Each assignment comprises the following elements:

- Jupyter Notebooks implementing the model and training pipeline.
- Supporting scripts or utility functions if needed. 

---

## Technical Stack

Developed in Python 3.11 using:

- **Deep learning:** `TensorFlow`, `PyTorch`
- **Time series & sequence models:** `LSTM`, `Transformer`, `Seq2Seq`
- **Visualization:** `matplotlib`, `seaborn`
- **Utilities:** `numpy`, `pandas`, `scikit-learn`, `yfinance`

> Some assignments may use additional specialized libraries such as `keras`, `scipy`, or `torch.nn.functional`.

---

## Overview of Assignments

The following section presents a concise overview of each task, highlighting its primary objective:

### Assignment 1 – *Extreme Learning Machines*
Implementation and comparison of a multilayer perceptron (`MLP`), a standard extreme learning machine (`ELM`), and a binary-weight ELM for emotion classification using `VQ-VAE` encoded inputs. The study evaluates the impact of different regularization strategies (none, Ridge, Lasso, ElasticNet) on the ELM’s output layer, using 12×12 integer matrices as input representations of facial expressions.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment1/output.png" alt=" Assignment 1" width="700"/>
</div>

### Assignment 2 – *Seq2Seq Prediction for Cryptocurrency Time Series*  
Implementation of a sequence-to-sequence (`Seq2Seq`) model with attention and teacher forcing to predict future values in multivariate time series of cryptocurrency prices. Using data from 7 cryptocurrencies (including `Bitcoin`) over 100 hourly intervals, the model forecasts the final segment of each series. Historical data is fetched via yahoo-finance and normalized using MinMax scaling. The model is trained for 300 epochs with LSTM layers of 1000 hidden units.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment2/output.png" alt=" Assignment 2" width="500"/>
</div>

### Assignment 3 – *Transformer Encoder for Time Series Forecasting*  
Implementation of a Transformer encoder model in `TensorFlow` to predict future cryptocurrency prices from past sequences. The model is built from scratch using `MultiHeadAttention` layers, with three encoding blocks and a latent dimension of 64. Two versions are explored: one using `logarithmic normalization`, and another using `MinMax scaling`. The model’s performance is compared against a naive baseline (no change in price) and the `Seq2Seq` architecture from Assignment 2.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment3/output.png" alt=" Assignment 3" width="500"/>
</div>

### Assignment 4 – *Transfer Learning and LoRA for Currency Forecasting* 
Extension of the Transformer model from Assignment 3 to a new dataset including exchange rates for multiple currencies and daily oil prices. Three strategies are compared: (1) training a new model from scratch, (2) full fine-tuning of the pretrained transformer, and (3) fine-tuning using Low-Rank Adaptation (`LoRA`) on affine layers. Multiple `LoRA` ranks are tested to evaluate efficiency vs. performance trade-offs.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment4/output.png" alt=" Assignment 4" width="500"/>
</div>

### Assignment 5 – *Diffusion Model with Transformer Sampler*  
Adaptation of a `DDIM-based diffusion model` by replacing the original `U-Net` architecture with a custom Transformer for the reverse sampling process. The architecture uses 12 attention heads across 8 layers, maintaining all other parameters from the original implementation (e.g., number of diffusion steps, embedding size, learning rate). The model is trained for 50 epochs due to high computational cost.

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment5/output.png" alt=" Assignment 5a" width="500"/>
</div>

<div align="center">
  <img src="https://github.com/ezautorres/Advanced-Deep-Learning-CIMAT/blob/main/assignment5/output2.png" alt=" Assignment 5b" width="500"/>
</div>

---

### Final Project – *State-Exchange Attention (SEA) for Physics Transformers*  
Investigation and replication of the SEA architecture proposed by Esmati et al. (2024), which integrates a novel State-Exchange Attention mechanism into transformer-based models for simulating PDE-governed physical systems. The SEA module enables dynamic cross-field communication between state variables such as velocity, pressure, and volume fraction, effectively reducing autoregressive rollout error. The full ViT-SEA framework achieves up to 91% error reduction compared to state-of-the-art baselines, demonstrating its capacity to capture complex spatiotemporal dynamics in computational fluid dynamics scenarios.  
For further details, see the original publication by [Esmati et al. (2024)](https://arxiv.org/abs/2403.04603).

---

## Learning Outcomes

- Built custom deep learning models for time series forecasting and generative modeling.
- Gained hands-on experience with transformer architectures, LSTMs, and ELMs.
- Explored fine-tuning strategies including full transfer learning and LoRA.
- Adapted diffusion models and autoregressive frameworks to novel architectures.
- Analyzed and visualized model performance across financial and physical domains.

---

## References

- Esmati, S., Gholami, A., & Mahoney, M. W. (2024).
  *State Exchange Attention for Physics Transformers*. arXiv:2403.04603.  
  [https://arxiv.org/abs/2403.04603](https://arxiv.org/abs/2403.04603)

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., & Chen, W. (2021).
  *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.  
  [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

---

## 📫 Contact

- 📧 Email: ezau.torres@cimat.mx  
- 💼 LinkedIn: [linkedin.com/in/ezautorres](https://linkedin.com/in/ezautorres)