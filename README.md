# Automatic Sleep Disorder Classification

This repository contains the implementation, experiments, and analysis for the **Automatic Sleep Disorder Classification** project, conducted at **Maastricht University**.  
The project investigates the effectiveness of **machine learning and foundation model representations** for diagnosing sleep disorders from polysomnography (PSG) data in the presence of **severe class imbalance and limited clinical data**.

---

## Motivation

Sleep disorders affect a significant portion of the global population and are associated with cardiovascular disease, metabolic dysfunction, depression, and cognitive decline.  
Despite their prevalence, diagnosis still relies largely on **manual scoring of PSG recordings by trained experts**, which is:

- Labor-intensive  
- Time-consuming  
- Prone to inter-scorer variability  

This project explores whether **automatic, data-driven methods** can assist or improve sleep disorder diagnosis under realistic clinical constraints.

---

## Problem Setting

The task is formulated as a **multi-class night-level classification problem**, where each full-night PSG recording is assigned a sleep disorder label.

Key challenges:
- Very **small dataset size**
- **Severe class imbalance**
- High-dimensional and heterogeneous physiological signals
- Risk of data leakage if subject-level separation is not enforced

---

## Dataset

- **CAP Sleep Database** (PhysioNet)
- 82 full-night PSG recordings
- Multiple physiological modalities (EEG, ECG, respiratory signals)
- After filtering rare categories, 5 target classes remain:
  - Healthy Control  
  - NFLE (Nocturnal Frontal Lobe Epilepsy)  
  - RBD (REM Behavior Disorder)  
  - PLMD (Periodic Limb Movement Disorder)  
  - Insomnia  

Class distribution is highly skewed, with the largest class containing more than four times the samples of the smallest class.

---

## Feature Representations

Two fundamentally different representations are evaluated:

### 1. Statistical Features
Handcrafted features extracted from PSG signals, including:
- Time-domain statistics
- Frequency-domain characteristics
- Aggregated nightly descriptors

These features serve as **classical machine-learning baselines**.

### 2. SleepFM Embeddings
- Pre-trained multimodal foundation model
- Trained on over 100,000 hours of PSG data
- Used in a **frozen feature-extraction setup**
- Provides compact, high-level representations of sleep physiology

---

## Models

The following models are evaluated across both feature representations:

- Random Forest
- k-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- Logistic Regression (primarily for embedding validation)

Model selection focuses on robustness and interpretability rather than deep end-to-end training, given the dataset size.

---

## Handling Class Imbalance

Multiple imbalance mitigation strategies are systematically studied:

- Class weighting
- Random oversampling
- Focal loss
- Adaptive noise injection scaled by feature variance

These methods are evaluated to determine which strategies are most effective under extreme imbalance.

---

## Evaluation Protocol

To ensure realistic and fair evaluation:
- **Subject-aware splitting** is enforced to avoid data leakage
- Performance is reported using **imbalance-aware metrics**:
  - Macro-F1 score
  - AUROC
  - AUPRC
- Results are averaged across validation folds where applicable

Accuracy alone is intentionally avoided due to its misleading nature in imbalanced settings.

---

## Key Results

- **SleepFM embeddings combined with Random Forest** achieve strong performance without explicit imbalance handling:
  - AUROC ≈ 0.87
  - Macro-F1 ≈ 0.41
- Classical statistical baselines require careful imbalance mitigation to remain competitive
- Oversampling-based methods consistently outperform loss-based techniques
- Foundation model representations show improved **class separability and robustness** in low-data regimes

---

## Contributions

- Empirical comparison between **foundation models and classical ML** under realistic clinical constraints
- Systematic analysis of **imbalance handling strategies**
- Reproducible preprocessing and evaluation pipeline
- Insight into the limits and potential of representation learning for sleep medicine

---

## Technologies

- Python
- NumPy
- scikit-learn
- SleepFM embeddings
- Machine learning for time-series and biomedical data

---

## Authors

Group 13  
- Popa Ștefan-Andrei  
- Eduard Levinschi  
- Aedem Bangerter  
- Daniel Skuczi  
- Yannick Brackelaire  
- Jonas Daukša  

Supervised at **Maastricht University**, Faculty of Science and Engineering.

---

## Disclaimer

This project is intended **solely for research and educational purposes**.  
It does not constitute a clinical diagnostic system and should not be used for medical decision-making.
