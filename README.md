# Fraud Detection System

A machine learning pipeline that detects fraudulent credit card transactions from highly imbalanced transaction data, built to explore real-world trade-offs between catching fraud and minimizing false alarms.

## Project Goal

Credit card fraud is rare — in this dataset, only 0.17% of transactions are fraudulent (492 out of 284,807). A naive model can hit 99.8% accuracy by simply predicting "not fraud" every time, which makes this a genuinely hard imbalanced classification problem. This project builds, evaluates, and compares two models to detect fraud while managing the trade-off between **recall** (catching actual fraud) and **precision** (avoiding false alarms).

Beyond classification, the project is being extended to treat fraud detection as an **economic decision problem** — since in production, a missed fraud and a false alarm carry very different real-world costs.

## Key Features

- End-to-end pipeline: data loading → preprocessing → training → evaluation → prediction
- Handles severe class imbalance using `class_weight="balanced"`
- Compares two models (Logistic Regression vs. Random Forest) with a clear precision/recall trade-off analysis
- Evaluation focused on precision, recall, F1, and ROC-AUC — not accuracy, since accuracy is misleading here
- **Risk scoring**: continuous fraud probability output instead of a fixed binary prediction, enabling downstream cost-based decisions
- **Evidence-based cost framework**: quantifies the asymmetric cost of false negatives vs. false positives using values grounded in published industry and academic sources, not arbitrary assumptions
- **Expected loss optimization**: identifies the decision threshold that minimizes total financial cost, not just maximizes F1
- Modular, testable codebase with unit tests for preprocessing, prediction, risk scoring, and cost analysis

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

**Risk score separation:** Random Forest's predicted probabilities separate the two classes clearly — normal transactions average a risk score of 0.0009, while fraud transactions average 0.77 (median 0.94). This confirms the model has strong discriminative power, which matters for the threshold optimization work below.

## Cost-Sensitive Extension (In Progress)

Standard classification metrics (precision, recall, F1) treat every false positive and every false negative as equally costly. In reality, missing a $500 fraud transaction is a very different loss than flagging a legitimate $20 purchase for review. This section reframes the problem economically.

**Approach:**
- **Risk scoring** (`src/risk_scoring.py`): replaces binary 0/1 output with a continuous fraud probability, enabling downstream cost-based decisions instead of a fixed 0.5 cutoff.
- **Cost framework** (`src/cost_analysis.py`): defines the asymmetric cost of errors using empirically grounded values:
  - **False Negative cost ≈ $108.62** — the average dollar amount of missed fraud transactions in the test set (computed directly from data, not assumed).
  - **False Positive cost = $25.00** — a conservative industry benchmark for manual alert review cost at mid-size financial institutions.
  - **Resulting cost ratio ≈ 4.3:1** — a missed fraud is, on average, about 4.3x more costly than a false alarm in this dataset.

**Sources informing this approach:**
- Chen et al. (2026), *A Regulatory Governance Framework for AI-Driven Financial Fraud Detection in U.S. Banking*, [arXiv:2605.04076](https://arxiv.org/pdf/2605.04076) — net-savings formula using empirical transaction value rather than assumed flat costs.
- FluxForce (2024), *False Positive Rates in Transaction Monitoring*, citing LexisNexis Risk Solutions' 2024 compliance cost survey of 1,000+ institutions — [source](https://www.fluxforce.ai/statistics/false-positive-rates-transaction-monitoring).
- EAI Endorsed Transactions (2026), *Enhancing Credit Card Fraud Detection under Severe Class Imbalance using Cost-Sensitive Learning and Threshold Optimization* — a benchmark study using this same dataset, reducing false positives from 21 to 13 via threshold optimization ([source](https://publications.eai.eu/index.php/ismla/article/view/12078)).

**Result:** Optimizing for expected loss instead of F1 shifts the optimal decision threshold from the default 0.5 down to 0.21 — the system becomes more aggressive about flagging fraud, since missing fraud is ~4.3x costlier than a false alarm. This trade-off (11 missed fraud vs. 19, at the cost of 28 false alarms vs. 7) reduces total expected loss by 15.4% ($2,238.81 → $1,894.84) compared to the default threshold.

**Completed:** risk scoring, evidence-based cost framework, and expected loss optimization across thresholds (see `src/expected_loss.py`).

**Planned next steps:** a 3-tier decision layer (approve / manual review / block) built on top of the optimal threshold, and a full-dataset simulation quantifying total cost impact if this system were deployed.

## Project Structure

```
fraud-detection-system/
├── data/               # raw and processed datasets (not committed)
├── notebooks/          # exploratory data analysis
├── src/                # core pipeline modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── risk_scoring.py
│   ├── cost_analysis.py
│   └── expected_loss.py
├── models/             # saved model artifacts (not committed)
├── tests/              # unit tests
└── app.py              # optional demo interface
```

## How to Run

1. **Clone the repository**

```bash
git clone https://github.com/Raditxt/fraud-detection-system.git
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
python -m src.train             # trains and saves both models
python -m src.evaluate          # prints precision/recall/F1/ROC-AUC for both models
python -m src.predict           # runs a sample prediction
python -m src.risk_scoring      # shows risk score distribution by class
python -m src.cost_analysis     # prints the cost matrix (FN vs. FP)
python -m src.expected_loss     # finds the cost-minimizing threshold
```

6. **Run tests**

```bash
python -m pytest tests/ -v
```

## Challenges & Learnings

- **Class imbalance** was the core challenge — with fraud at just 0.17% of the data, accuracy became a meaningless metric. Switching evaluation focus to precision/recall/F1 was necessary to actually measure model usefulness.
- **Precision vs. recall trade-off**: Logistic Regression and Random Forest optimize this trade-off very differently. This project made it clear that "best model" depends on business priorities — is it worse to miss fraud, or to annoy customers with false flags?
- **Stratified splitting** was essential; without it, a random train/test split risks leaving too few (or zero) fraud examples in the test set, making evaluation unreliable.
- **Metrics alone don't make decisions**: precision and recall describe model behavior, but they don't say what to *do* about it. Building the cost framework required grounding assumptions in real industry data rather than picking arbitrary cost values, since arbitrary costs would make any resulting "optimal threshold" meaningless.

## Future Improvements

- Simulate the decision system across the full test set to quantify total cost reduction vs. the default 0.5 threshold baseline
- Add SHAP-based interpretability to explain individual fraud predictions
- Add a simple Streamlit demo (`app.py`) to interactively test transactions
- Try gradient boosting models (XGBoost/LightGBM) for comparison, evaluated by expected loss rather than accuracy alone