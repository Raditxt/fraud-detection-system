# Fraud Detection System

A machine learning pipeline that detects fraudulent credit card transactions from highly imbalanced transaction data, built to explore real-world trade-offs between catching fraud and minimizing false alarms.

## Project Goal

Credit card fraud is rare — in this dataset, only 0.17% of transactions are fraudulent (492 out of 284,807). A naive model can hit 99.8% accuracy by simply predicting "not fraud" every time, which makes this a genuinely hard imbalanced classification problem. This project builds, evaluates, and compares two models to detect fraud while managing the trade-off between **recall** (catching actual fraud) and **precision** (avoiding false alarms).

## Key Features

- End-to-end pipeline: data loading → preprocessing → training → evaluation → prediction
- Handles severe class imbalance using `class_weight="balanced"`
- Compares two models (Logistic Regression vs. Random Forest) with a clear precision/recall trade-off analysis
- Evaluation focused on precision, recall, F1, and ROC-AUC — not accuracy, since accuracy is misleading here
- Modular, testable codebase with unit tests for preprocessing and prediction logic

## Tech Stack

- **Python 3.11+**
- **pandas** / **numpy** — data manipulation
- **scikit-learn** — modeling and evaluation
- **joblib** — model persistence
- **pytest** — unit testing

## Results

| Model               | Precision (Fraud)  | Recall (Fraud)  | F1 (Fraud) | ROC-AUC |
|---------------------|--------------------|-----------------|------------|---------|
| Logistic Regression | 0.06               | 0.92            | 0.11       | 0.9722  |
| Random Forest       | 0.92               | 0.81            | 0.86       | 0.9518  |

**Takeaway:** Logistic Regression catches more fraud (92% recall) but generates a large number of false positives (1,389 normal transactions flagged), which would be costly to review manually. Random Forest is far more precise (92%) with only 7 false positives, making it more practical for real-world deployment — at the cost of missing slightly more fraud cases (19 vs. 8 missed).

## Project Structure
fraud-detection-system/
├── data/               # raw and processed datasets (not committed)
├── notebooks/          # exploratory data analysis
├── src/                # core pipeline modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/             # saved model artifacts (not committed)
├── tests/              # unit tests
└── app.py              # optional demo interface

## How to Run

1. **Clone the repository**
```bash
   git clone https://github.com/<your-username>/fraud-detection-system.git
   cd fraud-detection-system
```

2. **Set up a virtual environment**
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Download the dataset**
   Get `creditcard.csv` from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/raw/`.

5. **Run the pipeline**
```bash
   python -m src.train       # trains and saves both models
   python -m src.evaluate    # prints precision/recall/F1/ROC-AUC for both models
   python -m src.predict     # runs a sample prediction
```

6. **Run tests**
```bash
   python -m pytest tests/ -v
```

## Challenges & Learnings

- **Class imbalance** was the core challenge — with fraud at just 0.17% of the data, accuracy became a meaningless metric. Switching evaluation focus to precision/recall/F1 was necessary to actually measure model usefulness.
- **Precision vs. recall trade-off**: Logistic Regression and Random Forest optimize this trade-off very differently. This project made it clear that "best model" depends on business priorities — is it worse to miss fraud, or to annoy customers with false flags?
- **Stratified splitting** was essential; without it, a random train/test split risks leaving too few (or zero) fraud examples in the test set, making evaluation unreliable.

## Future Improvements

- Experiment with SMOTE (oversampling) instead of `class_weight` to compare imbalance-handling strategies
- Tune classification threshold on Random Forest to improve recall without sacrificing too much precision
- Add a simple Streamlit demo (`app.py`) to interactively test transactions
- Try gradient boosting models (XGBoost/LightGBM) for comparison