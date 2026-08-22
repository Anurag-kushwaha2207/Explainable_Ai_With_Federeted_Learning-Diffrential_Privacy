# Privacy-Preserving Federated Learning with Differential Privacy for IoT Health Monitoring

A privacy-preserving machine learning framework for **IoT-based health monitoring** that combines **Differential Privacy (DP)**, **Federated Learning (FL)**, and **Explainable AI (SHAP)**.

The system enables multiple edge devices, such as smart health monitoring devices, to collaboratively train a global machine learning model without sharing raw patient records with a central server.

---

## Overview

Healthcare IoT devices continuously generate sensitive patient data such as physiological measurements and health-related parameters.

This project addresses the privacy challenge by keeping patient data on local client devices while allowing collaborative machine learning through Federated Learning.

The framework integrates:

- **Differential Privacy**
- **Federated Learning**
- **Explainable AI (SHAP)**
- **Non-IID Data Distribution**
- **Privacy-Preserving Feature Statistics**
- **Leakage-Safe Evaluation**
- **Health Event Classification**

---

# 1. Differential Privacy

The framework uses **Single-Release Laplace Output Differential Privacy** to protect the final model output.

Feature norm clipping is applied before adding Laplace noise.

A unified clipping constant is used:

$$
R_{\text{clip}} = 1.5
$$

The privacy mechanism uses a configurable privacy budget $\epsilon$.

### Single-Release DP

Intermediate federated learning rounds perform clean model aggregation.

Differential Privacy noise is introduced only during the final release:

$$
\Delta W = \frac{2 \cdot R_{\text{clip}}}{N \cdot \alpha}
$$

$$
W_{\text{private}}
=
W_{\text{local}}
+
\text{Laplace}
\left(
0,
\frac{\Delta W}{\epsilon}
\right)
$$

Smaller values of $\epsilon$ provide stronger privacy but introduce higher noise.

---

# 2. Federated Learning

The system uses a **client-server Federated Learning architecture**.

Multiple client devices independently train local models using their local health data.

Raw patient records remain on the client side.

Only model information is communicated to the central server for aggregation.

### Federated Aggregation

The central server uses **FedAvg (Federated Averaging)** to combine the client models and create a global model.

The system supports a **non-IID Dirichlet data distribution** across three federated clients.

---

# 3. Privacy-Preserving Federated Feature Statistics

To avoid sharing raw patient data while maintaining consistent feature normalization, clients share only scalar statistics:

- Mean $(\mu_i)$
- Variance $(\sigma_i^2)$
- Sample count $(N_i)$

The federated mean is calculated as:

$$
\mu_{\text{fed}}
=
\frac{\sum N_i\mu_i}
{\sum N_i}
$$

The federated variance is calculated as:

$$
\sigma_{\text{fed}}^2
=
\frac{
\sum
[
N_i\sigma_i^2
+
N_i(\mu_i-\mu_{\text{fed}})^2
]
}
{\sum N_i}
$$

The server broadcasts the aggregated statistics back to clients so that local `StandardScaler` instances can use consistent normalization parameters.

Raw patient records are never required by the central server.

---

# 4. Leakage-Safe Evaluation

The evaluation pipeline separates real clinical records from synthetic training data.

The `train_test_split` operation is performed exclusively on:


5. System Architecture

The system consists of multiple edge clients and a central aggregation server.
is_synthetic == False
                    IoT Health Devices
                           |
          +----------------+----------------+
          |                |                |
       Client 1         Client 2         Client 3
          |                |                |
     Local Training   Local Training   Local Training
          |                |                |
          +----------------+----------------+
                           |
                           v
                  Central Server
                           |
                     FedAvg Aggregation
                           |
                           v
                    Global Model
                           |
                +----------+----------+
                |                     |
                v                     v
       Differential Privacy      SHAP Explainability
                |                     |
                v                     v
        Private Model Output   Interpretable Predictions
        
### STEP 6 — Experimental Benchmarks

```markdown
# 6. Experimental Benchmarks

The system was evaluated on **6,000 health records** distributed across **3 federated clients** using a non-IID Dirichlet distribution with:

$$
\alpha = 1.0
$$

The evaluation test set contains **100% real records**.
```
## Global & Personalized Model Performance
```
| Command | Privacy Configuration | Global Test Accuracy | Multi-Class ROC-AUC | Avg Personalized Accuracy | Accuracy Gap vs Baseline |
|---|---|---:|---:|---:|---:|
| `python server.py --no-dp` | Non-Private Baseline | **97.18%** | **0.9959** | **96.98%** | 0.00% |
| `python server.py -e 1.0` | DP Single-Release ($\epsilon = 1.0$) | **97.18%** | **0.9959** | **96.98%** | **0.00%** |
| `python server.py -e 0.5` | DP Single-Release ($\epsilon = 0.5$) | **96.96%** | **0.9959** | **96.98%** | **0.22%** |
# 7. Confusion Matrix

Results for the high-noise privacy configuration:

$$
\epsilon = 0.5
$$

```text
Actual / Predicted   Normal     Mild     Moderate     Severe
----------------------------------------------------------------
Normal               300        0        0             0
Mild Event             1       48        2             9
Moderate Event         0        0       53             0
Severe Event           0        2        0            46
```

### STEP 8 — Visualizations

```markdown
# 8. Architectural & Experimental Visualizations

## Feature Importance

<img width="3600" height="3000" alt="Feature Importance" src="https://github.com/user-attachments/assets/e86fd312-c50b-4eba-8bae-81075cfe0b12" />

## Accuracy Progression

<img width="2550" height="1800" alt="Accuracy Progression" src="https://github.com/user-attachments/assets/1b4dab82-711a-427e-aa47-b11791a9e630" />
```
# 9. Repository Structure

```text
├── README.md
├── LICENSE
├── run.txt
│
├── server.py
├── split_data.py
├── machine.py
├── machine1.py
├── machine2.py
├── machine3.py
│
├── compare_datasets.py
├── explain.py
├── encoding_mappings.json
│
├── Federated-data.xlsx
├── health_data_balanced_after_overfitting.xlsx
│
├── plot_accuracy.png
├── plot_dp_laplace.png
├── plot_dataset_sizes.png
├── plot_confusion_matrix.png
└── plot_feature_importance.png
```

### STEP 10 — Key Files

```markdown
# 10. Key Files

| File | Description |
|---|---|
| `server.py` | Central server for federated aggregation and model evaluation |
| `split_data.py` | Creates non-IID client data partitions |
| `machine.py` | Client training and evaluation runner |
| `machine1.py` | Federated Client 1 implementation |
| `machine2.py` | Federated Client 2 implementation |
| `machine3.py` | Federated Client 3 implementation |
| `compare_datasets.py` | Dataset comparison and validation |
| `explain.py` | Explainable AI and SHAP analysis |
| `encoding_mappings.json` | Feature encoding mappings |
| `Federated-data.xlsx` | Federated dataset |
| `health_data_balanced_after_overfitting.xlsx` | Main health telemetry dataset |
| `run.txt` | Verified experimental execution log |
| `plot_accuracy.png` | Accuracy progression across communication rounds |
| `plot_dp_laplace.png` | Laplace noise scale visualization |
| `plot_dataset_sizes.png` | Client dataset distribution |
| `plot_confusion_matrix.png` | Global model confusion matrix |
| `plot_feature_importance.png` | Feature importance visualization |
```
# 11. How to Run

## Step 1 — Partition Data

Run the data partitioning script:

```bash
python split_data.py
python server.py -e 1.0
python server.py -e 0.5
python server.py --no-dp
machine server.py
```

### STEP 12 — Explainable AI with SHAP

```markdown
# 12. Explainable AI with SHAP

The system integrates **SHAP (SHapley Additive exPlanations)** to improve the interpretability and transparency of machine learning predictions.

SHAP analyzes how individual health-related features contribute to model predictions and helps explain the decisions made by the trained model.

## Key Capabilities

- **Global Feature Importance** — Identifies the most influential health features across predictions.
- **Individual Prediction Explanations** — Explains why the model makes a specific prediction.
- **Influential Health Parameters** — Highlights health parameters that strongly affect model decisions.
- **Transparent Model Interpretation** — Provides interpretable insights into model predictions.
```
# 13. Key Technologies

- Python
- Federated Learning
- FedAvg
- Differential Privacy
- Laplace Mechanism
- Explainable AI
- SHAP
- Machine Learning
- IoT Health Monitoring
- Non-IID Data Distribution
- Privacy-Preserving Data Processing
- Pandas
- NumPy
- Scikit-learn

  # 14. Project Goals
```
The primary goals of this project are:

1. Protect sensitive health data during collaborative machine learning.
2. Keep raw patient records on local edge devices.
3. Enable collaborative model training using Federated Learning.
4. Apply Differential Privacy to protect the final model output.
5. Reduce evaluation leakage through real-data-only testing.
6. Provide interpretable model predictions using SHAP.
7. Maintain high classification performance under privacy constraints.
```
# 15. Author
```
**Anurag Singh**

B.Tech Computer Science Engineering

**Areas of Interest:**
- Federated Learning
- Differential Privacy
- Explainable AI
- Machine Learning
- IoT Health Monitoring
