
# Differential Evolution for Neural Network Optimization

## 📌 Project Overview

This project explores using **Differential Evolution (DE)** as an alternative to traditional backpropagation for training neural networks. The algorithm evolves neural network weights for classification tasks, particularly focusing on the **MNIST handwritten digit dataset**. The core objective is to analyze the performance, stability, and convergence of DE when compared to gradient-based training.

---

## 🚀 Key Features

- Initialization of population with random or Gaussian noise
- Fitness evaluation based on validation accuracy
- Supports multiple mutation strategies: `rand_1`, `best_1`
- Crossover options: binomial and exponential
- Selection mechanisms: greedy, tournament, crowding
- Tracks convergence metrics and enables early stopping

---

## 🧠 Similar Systems

- **NEAT** (Neuroevolution of Augmenting Topologies)
- **Google AutoML**
- **TPOT**
- **EvoNet**

---

## 📚 Literature References

- Storn & Price (1997): *Differential Evolution - A Simple and Efficient Heuristic for Global Optimization*
- Das & Suganthan (2011): *Survey of DE Algorithms*
- Pant et al. (2011): *Recent DE Variants*
- Salomon & Rodoplu (2003): *Training Neural Networks Using DE*
- Yao et al. (1999), Mezura-Montes & Coello Coello (2011)

(Full references in the PDF or see DOI links at the bottom of this README)

---

## 📊 Dataset Used

- **MNIST Handwritten Digits**
- Source: [Kaggle MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- Description: 70,000 grayscale images (28x28), labeled 0–9
- Preprocessing: Normalized pixel values, one-hot encoding if required

---

## 🧪 Algorithm & Experiment Setup

### Algorithm Steps

1. **Initialize population** of flattened NN weight vectors
2. **Mutation**: Apply `rand_1` or `best_1`
3. **Crossover**: Binomial or Exponential
4. **Selection**: Greedy, Tournament, or Crowding
5. **Repeat** for several generations while tracking performance

### Example Run

- Random Seed: 115000  
- Population Size: 20  
- Mutation Factor (F): 0.5  
- Crossover Rate (CR): 0.7  
- Generations: 1500  
- Mutation Strategy: `rand_1`  
- Crossover Strategy: `binomial`  
- Selection Strategy: `select_better`  
- Initialization Strategy: `random`

---

## 🛠 Development Environment

- **Language**: Python  
- **Frameworks**: TensorFlow 2.x, NumPy, Keras  
- **Platform**: Windows with NVIDIA GPU  
- **Editor**: Visual Studio Code  

### Project Files

- `de_algorithms.py`: Includes the Differential Evolution algorithm implementation
- `model_utils.py`: Contains model creation and weight handling utilities
- `main.py`: Runs the DE logic and experimentation loop

---

## 🔗 DOI Links to Key References

- [Yao et al., 1999](https://doi.org/10.1109/4235.771163)
- [Storn & Price, 1997](https://doi.org/10.1023/A:1008202821328)
- [Pant et al., 2011](https://doi.org/10.1007/s00500-010-0641-4)
- [Das & Suganthan, 2011](https://doi.org/10.1109/TEVC.2010.2059031)
- [Salomon & Rodoplu, 2003](https://doi.org/10.1109/IJCNN.2003.1223514)
- [Mezura-Montes & Coello Coello, 2011](https://doi.org/10.1016/j.swevo.2011.10.001)
