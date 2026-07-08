# Sensor Data Classification: CEIP-DS-JECRC

A machine learning solution for binary sensor data classification using **LightGBM** with custom feature engineering and threshold optimization for highly imbalanced datasets.

---

## 🚀 Overview

This project predicts binary anomaly labels from multivariate sensor data (`X1`–`X5`). The model captures temporal patterns through handcrafted features and improves classification performance by optimizing the prediction threshold instead of using the default value.

### Features

* Time-series feature engineering
* LightGBM classifier
* Handling of imbalanced classes using `scale_pos_weight`
* Chronological train-validation split
* Threshold optimization to maximize F1-score

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/alokzef/Kaggle-CEIP-DS-JECRC.git
cd Kaggle-CEIP-DS-JECRC
```

### 2. (Optional) Create a Virtual Environment

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

Place all dataset files inside the `/kaggle/input/competitions/ceip-ds-jecrc` folder.

```
/kaggle/input/competitions/ceip-ds-jecrc/
├── train.csv
├── test.csv
└── sample_submission.csv
```

---

## 🛠️ Feature Engineering

To improve the model's predictive power, the following transformations were applied:
* **Time Features:** Extracted `hour`, `day_of_week`, and `is_weekend` from timestamps.
* **Sensor Aggregates:** Computed row-wise `mean`, `std`, and `max` across all sensors.
* **Windowing & Smoothing:** * **Rolling Mean & Variance:** Captured local trends and volatility.
    * **EWMA:** Exponentially Weighted Moving Averages for noise reduction.
    * **Differencing:** Calculated step-to-step changes ($X_t - X_{t-1}$).
* **Signal Ratios:** Created $X1/X2$ and $X3/X4$ interaction features.

---

## 📉 Methodology

### Validation

A **73/27 chronological split** was used to prevent future data leakage and better simulate real-world forecasting.

### Feature Scaling

Features were standardized using **StandardScaler**.

### Model

* LightGBM (`LGBMClassifier`)
* `scale_pos_weight` used for class imbalance handling

### Threshold Optimization

Instead of the default probability threshold of **0.5**, multiple thresholds were evaluated to maximize the validation **F1-score**.

**Best threshold:** **0.97**

---

## 📊 Results

| Metric              |  Value |
| ------------------- | -----: |
| Validation F1-Score | ~0.618 |
| Optimal Threshold   |   0.97 |
| Estimators          |    250 |
| Learning Rate       |   0.03 |
| Max Depth           |      5 |

---

## 📁 Project Structure

```
.
├── dataset/
├── notebookCEIP_DS_JECRC.ipynb
├── scriptCEIP_DS_JECRC.py
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the Python script:

```bash
python scriptCEIP_DS_JECRC.py
```

Or open the notebook:

```
notebookCEIP_DS_JECRC.ipynb
```

---

## 🛠️ Built With

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM

---

## 📄 License

This project was developed for the **CEIP-DS-JECRC** machine learning competition. The dataset is subject to the competition's terms and conditions.
