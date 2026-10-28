# Signal Hunters — Round 2 (Hack4Hack ML Challenge 2026)

**Team Members:** Yash Ghotekar, Paras Medpalliwar, Soham Sirpurwar  
**Best Leaderboard Macro F1 Score:** `0.7896`  
**Evaluation Metric:** Macro F1 Score  

---

## 📌 Project Overview
This repository contains the end-to-end signal classification pipeline developed for **Round 2 of the Signal Hunters ML Challenge on Hack4Hack**.

Our solution leverages deep 1D Convolutional Neural Networks (CNN v4) optimized for time-series / signal patterns, combined with robust preprocessing, feature extraction, and post-training threshold tuning to maximize the Macro F1 score across test samples.

---

## 📁 Repository Structure

```text
├── pipeline_v2/
│   ├── train_cnn_v4.py           # Final & best-performing 1D CNN model training
│   ├── predict_v3.py             # Final inference pipeline generating submission.csv
│   ├── detailed_error_analysis.py# Post-training evaluation and error analysis
│   ├── train_cnn.py              # Baseline CNN training script
│   ├── train_v3.py               # Iteration v3 training pipeline
│   └── README.md                 # Pipeline documentation
├── starter_notebook.ipynb        # Exploratory Data Analysis & initial experiments
├── submission.csv                # Final predictions (columns: "id", "label")
├── requirements.txt              # Environment dependencies
└── README.md                     # Main documentation
```

---

## ⚙️ Requirements & Environment Setup

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

*(Key libraries: `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`)*

---

## 🚀 How to Reproduce Results

### 1. Model Training
To train the final 1D CNN architecture:
```bash
python pipeline_v2/train_cnn_v4.py
```
* **Loss Function:** Cross-Entropy / Focal Loss tuned for class imbalance
* **Optimizer:** AdamW with cosine annealing learning rate scheduler
* **Validation Strategy:** Stratified K-Fold / Stratified Split evaluating Macro F1

### 2. Generate Final Predictions
To generate the final `submission.csv` on the test dataset:
```bash
python pipeline_v2/predict_v3.py
```
* Reads test signal samples.
* Performs exact feature normalization matching training.
* Outputs `submission.csv` with the exact required format:
  ```csv
  id,label
  0,1
  1,0
  ...
  ```

### 3. Error Analysis (Optional)
To inspect misclassified samples and confidence distributions:
```bash
python pipeline_v2/detailed_error_analysis.py
```

---

## 🏆 Key Improvements & Pipeline Progression
* **v1 - Baseline:** Initial tabular/heuristic models (`~0.47` Macro F1).
* **v2 - Feature Engineering:** Signal statistical and frequency domain features (`~0.64` - `0.71` Macro F1).
* **v3 - CNN Modeling:** 1D ConvNet capturing temporal signal patterns (`0.7784` - `0.7843` Macro F1).
* **v4 - Final Optimized Architecture (`train_cnn_v4.py`):** Deep 1D CNN with residual connections, batch normalization, dropout regularization, and calibrated decision boundaries, achieving our peak score of **`0.7896 Macro F1`**.
