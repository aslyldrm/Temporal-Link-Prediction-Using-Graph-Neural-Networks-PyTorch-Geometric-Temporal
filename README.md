# Temporal Link Prediction Using Graph Neural Networks

## 📌 Project Overview

Temporal link prediction addresses the problem of predicting whether a connection (edge) will form between two nodes in a graph within a specific time window. This is crucial for dynamic networks such as:

- Social media event forecasting
- Recommender systems
- E-commerce demand prediction

We implemented and compared three dynamic GNN architectures:

- **Temporal Graph Network (TGN)**
- **Temporal Graph Attention Network (TGAN)**
- **Dynamic Self-Attention Network (DySAT)**

The project is developed using [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal).

---

## 📂 Datasets

Datasets are from the [WSDM Cup 2022 Temporal Link Prediction Challenge](https://github.com/dglai/WSDM2022-Challenge):

- **Dataset A:** Event-based dynamic graph
- **Dataset B:** User-item interaction graph *(not used in this report)*

Each dataset contains:
- `edges_train.csv`
- `node_features.csv`
- `edge_type_features.csv`
- `input.csv` for test queries

---

## ⚙️ Model Architecture

| Model | Time Handling | Core Mechanism | Best For |
|-------|----------------|----------------|----------|
| TGN   | Continuous     | Memory modules with message passing | Generalization |
| TGAN  | Continuous     | Temporal attention mechanism        | Balanced tasks |
| DySAT | Discrete       | Structural + temporal self-attention | Pattern-heavy datasets |

---

## 🧪 Training Setup

- **Hardware:** Google Colab (GPU)
- **Epochs:**
  - TGN: 8
  - TGAN: 15
  - DySAT: 88
- **Loss Function:** Binary Cross-Entropy
- **Evaluation Metrics:** AUC, Accuracy, F1 Score, Temporal Leakage Analysis

---

## 📈 Results Summary

| Model | AUC | Clean AUC | Accuracy | F1 Score |
|-------|-----|-----------|----------|----------|
| DySAT | 0.7555 | 0.5554 | 78.08% | 0.8726 |
| TGAN  | 0.6579 | 0.6597 | 77.11% | 0.8688 |
| TGN   | 0.7110 | 0.7139 | 76.84% | 0.8677 |

**Conclusion:**  
TGN demonstrated the best generalization on unseen future links. DySAT, while accurate on repeated patterns, underperformed on clean data. TGAN provided a good trade-off between both.

---

## 📚 References

- Rossi et al., *Temporal Graph Networks*, arXiv:2006.10637
- Xu et al., *Inductive Representation Learning on Temporal Graphs*, arXiv:2002.07962
- Sankar et al., *Dynamic Graph Representation Learning via Self-Attention*, WWW 2020
- [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
- [WSDM 2022 Challenge](https://github.com/dglai/WSDM2022-Challenge)

---

## 👩‍💻 Authors

- Aslı Yıldırım – `20190808039@ogr.akdeniz.edu.tr`
- Emine Side Duran – `20200808605@ogr.akdeniz.edu.tr`

---

## 📌 Note

This project focuses on **Dataset A**. For a complete implementation including Dataset B, see [WSDM Challenge GitHub](https://github.com/dglai/WSDM2022-Challenge).
