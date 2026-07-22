# Fraud Detection System

A machine learning pipeline that detects fraudulent credit card transactions from highly imbalanced transaction data, built to explore real‑world trade‑offs between catching fraud and minimising false alarms.

## Project Goal

Credit card fraud is rare — in this dataset, only 0.17% of transactions are fraudulent (492 out of 284,807). A naive model can hit 99.8% accuracy by simply predicting "not fraud" every time, which makes this a genuinely hard imbalanced classification problem. This project builds, evaluates, and compares multiple models to detect fraud while managing the trade‑off between **recall** (catching actual fraud) and **precision** (avoiding false alarms).

Beyond classification, the project extends into treating fraud detection as an **economic decision system** — since in production, a missed fraud, a false alarm sent to review, and a legitimate transaction wrongly blocked all carry very different real‑world costs. The system computes fraud risk, assigns transactions to a 3‑tier decision (approve / review / block), and simulates the total dollar cost of running it end‑to‑end.

## Key Features

- End‑to‑end pipeline: data loading → preprocessing → training → evaluation → prediction
- Handles severe class imbalance using `class_weight="balanced"`
- Compares three models (Logistic Regression, Random Forest, XGBoost) with a clear precision/recall trade‑off analysis
- Evaluation focused on precision, recall, F1, and ROC‑AUC — not accuracy, since accuracy is misleading here
- **Risk scoring**: continuous fraud probability output instead of a fixed binary prediction, enabling downstream cost‑based decisions
- **Evidence‑based cost framework**: quantifies the asymmetric cost of false negatives vs. false positives using values grounded in published industry and academic sources, not arbitrary assumptions
- **Expected loss optimisation**: identifies the decision threshold that minimises total financial cost, not just maximises F1
- **3‑tier decision layer**: approve / manual review / block, reducing manual review workload to a small fraction of transactions
- **End‑to‑end system simulation**: quantifies total dollar cost and savings versus a no‑detection baseline, run independently across all three models for a fair cost‑based comparison
- **Model interpretability (SHAP)**: explains global feature importance and individual predictions, supporting auditability and case‑level review
- Modular, testable codebase (29 unit tests) covering preprocessing, prediction, risk scoring, cost analysis, expected loss, decision tiers, and system simulation

## Tech Stack

- **Python 3.11+**
- **pandas** / **numpy** — data manipulation
- **scikit‑learn** — modelling and evaluation
- **XGBoost** — gradient boosting, adopted as final model
- **SHAP** — model interpretability
- **joblib** — model persistence
- **pytest** — unit testing

## Results

| Model               | Precision (Fraud)  | Recall (Fraud)  | F1 (Fraud) | ROC‑AUC |
|---------------------|--------------------|-----------------|------------|---------|
| Logistic Regression | 0.06               | 0.92            | 0.11       | 0.9722  |
| Random Forest       | 0.92               | 0.81            | 0.86       | 0.9518  |

**Takeaway:** Logistic Regression catches more fraud (92% recall) but generates a large number of false positives (1,389 normal transactions flagged), which would be costly to review manually. Random Forest is far more precise (92%) with only 7 false positives, making it more practical for real‑world deployment — at the cost of missing slightly more fraud cases (19 vs. 8 missed). Random Forest was initially used as the base model for the cost‑sensitive extension below; see the **model comparison by total system cost** section for why XGBoost was ultimately adopted as the final model instead.

**Risk score separation:** Random Forest's predicted probabilities separate the two classes clearly — normal transactions average a risk score of 0.0009, while fraud transactions average 0.77 (median 0.94). This confirms the model has strong discriminative power, which the threshold optimisation work below relies on.

## Cost‑Sensitive Extension

Standard classification metrics (precision, recall, F1) treat every false positive and every false negative as equally costly. In reality, missing a fraud transaction, sending a legitimate one to manual review, and auto‑blocking a legitimate transaction all have different, measurable dollar costs. This section reframes the problem economically instead of purely statistically.

### Cost framework

- **False Negative cost ≈ $108.62** — the average dollar amount of missed fraud transactions in the test set (computed directly from data, not assumed).
- **False Positive (review) cost = $25.00** — a conservative industry benchmark for manual alert review cost at mid‑size financial institutions.
- **Resulting cost ratio ≈ 4.3:1** — a missed fraud is, on average, about 4.3x more costly than a false alarm sent to review.

### Expected loss & threshold optimisation

Optimising for expected loss instead of F1 shifts the optimal decision threshold from the default 0.5 down to **0.21** — the system becomes more aggressive about flagging fraud, since missing fraud is costlier than a false alarm. This trade‑off (11 missed fraud vs. 19, at the cost of 28 false alarms vs. 7) reduces total expected loss by **15.4%** ($2,238.81 → $1,894.84) compared to the default threshold.

### 3‑tier decision layer

A single threshold is still a binary decision. In practice, institutions use tiered decisioning: very low risk is approved instantly, very high risk is blocked instantly (no human needed), and the ambiguous middle band goes to manual review — mirroring industry practice where mid‑size institutions target a 30‑50% false positive rate on *reviewed* alerts rather than reviewing everything ([Flagright, 2024](https://www.flagright.com/post/understanding-false-positives-in-transaction-monitoring), citing LexisNexis Risk Solutions).

Using the cost‑optimal threshold (0.21) as the review cutoff and the lowest threshold achieving ≥95% precision (0.61) as the block cutoff:

|   Tier  | Transactions | Actual Fraud | Fraud Rate |
|---------|--------------|--------------|------------|
| Approve |    56,847    |      11      |    0.02%   |
| Review  |      36      |      12      |   33.33%   |
| Block   |      79      |      75      |   94.94%   |

Only **36 of 56,962 transactions (0.06%)** require manual review, while still catching 87 of 98 fraud cases automatically (via review + block combined).

### System simulation: full pipeline impact

To answer the practical question — "if this system were deployed, what's the dollar impact?" — the full pipeline (risk scoring → cost framework → expected loss → 3‑tier decision layer) was simulated end‑to‑end on the test set.

| Scenario | Missed Fraud Loss | Review Cost | False‑Block Loss | **Total Cost** |
|---|---|---|---|---|
| Do‑nothing baseline (no detection) | $10,644.93 | $0.00 | $0.00 | **$10,644.93** |
| 3‑tier system (approve / review / block) | $1,826.50 | $900.00 | $2.76 | **$2,729.26** |

**Result: $7,915.67 saved (74.4% cost reduction)** versus having no fraud detection at all, while requiring manual review on only 0.06% of transactions.

**Methodology notes:**
- Missed‑fraud loss uses the actual dollar amount of each specific fraud case that slipped through, not an average — more precise than the expected‑loss approximation used for threshold selection.
- Review cost is a flat $25 per transaction sent to manual review.
- False‑block loss uses the transaction's own dollar amount as a conservative cost floor. Published research suggests the real‑world cost of false declines is substantially higher once lost customer lifetime value is included — merchants report losing $30‑75 for every $1 of fraud prevented, due to customers who don't return after a false decline ([Aite Group, 2019, cited by INETCO](https://www.inetco.com/why-false-declines-cost-you-more-than-fraud-and-what-to-do-about-it/); [Corgi Labs, 2026](https://dev.corgilabs.ai/insights/false-declines), citing Merchant Risk Council 2024). This project deliberately does not apply that multiplier, since it was measured in an e‑commerce checkout context that may not transfer directly to this dataset — the reported figure is a conservative lower bound, not the full real‑world cost.
- **Limitation:** the 4 false‑blocked transactions in this test set happened to be low‑value ($2.76 total), which kept false‑block loss minimal. This reflects this specific test split, not evidence that the system optimises for transaction value — the block threshold is chosen purely on precision, with no awareness of dollar amount.

### Model comparison by total system cost

The same cost‑sensitive pipeline (risk scoring → expected loss optimisation → 3‑tier decision) was run independently for all three models, to test whether higher classification metrics (F1, recall) actually translate into lower financial cost.

| Model | Missed Fraud Loss | Review Cost | False‑Block Loss | **Total Cost** | vs. Baseline |
|---|---|---|---|---|---|
| Logistic Regression | $1,832.72 | $850.00 | $27,603.15 | **$30,285.87** | **‑184.5%** (worse than no detection) |
| Random Forest | $1,826.50 | $900.00 | $2.76 | **$2,729.26** | +74.4% |
| **XGBoost** | $1,942.10 | $375.00 | $2.77 | **$2,319.87** | **+78.2% (best)** |

**Key finding:** Logistic Regression — despite having the highest fraud recall (92%) among all three models — produces a system that is financially *worse than having no fraud detection at all*. Its low precision (0.06) means the model cannot reliably distinguish fraud from normal transactions at any confidence level, so the auto‑block tier ends up rejecting a large volume of legitimate transactions ($27,603 in false‑block losses alone). This is direct, quantified evidence that **classification metrics and business outcomes are not the same thing** — a model can look strong on recall and still be unusable in a cost‑sensitive deployment.

XGBoost outperforms Random Forest by $409.39 (15% lower total cost), primarily by requiring less manual review ($375 vs. $900) while maintaining comparably low false‑block losses. XGBoost is adopted as the final model for this reason — not because of its F1 or ROC‑AUC, but because it minimises total deployment cost.

### Interpretability (SHAP)

Because `V1`‑`V28` are PCA‑transformed for privacy, their real‑world meaning is unknown — but SHAP still explains *which* of them drive each prediction and by how much, which matters for case‑level review and auditability.

**Global feature importance** (mean absolute SHAP value across a 2,000‑transaction sample, XGBoost model):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | V14 | 2.95 |
| 2 | V4  | 1.77 |
| 3 | V12 | 1.04 |
| 4 | V10 | 0.86 |
| 5 | V11 | 0.78 |

**Cross‑validation with EDA:** 3 of the top 5 SHAP features (`V14`, `V12`, `V10`) also appear among the top 5 features most correlated with fraud in the initial EDA (`notebooks/01_eda.ipynb`), despite being derived from completely different methods — simple linear correlation vs. SHAP values from a non‑linear tree ensemble. This agreement is evidence that `V14` in particular is a robust, method‑independent fraud signal in this dataset, not a modelling artifact specific to one technique.

**Example: explaining a single fraud transaction.** For the highest‑risk transaction in the sample (correctly identified as fraud), `V14 = -3.84` was the largest single contributor to its fraud score (SHAP value +5.70), followed by `V4 = 7.04` (+2.22) and `V10 = -20.95` (+2.09). This is the kind of transaction‑level explanation a fraud analyst or auditor would need — not just "the model said fraud," but which specific signals drove that call and how strongly.

### Robustness (Exploratory Stress Testing)

Unlike the cost framework above, no external quantitative benchmark was found for "realistic" fraud‑ratio drift or feature noise levels in production systems — these tests are exploratory robustness checks against this specific model and dataset, not validated against industry research.

**Feature noise injection** (Gaussian noise scaled to each feature's own standard deviation, threshold held fixed at 0.21):

| Noise Level | Precision | Recall |
|---|---|---|
| Baseline (no noise) | 0.847 | 0.847 |
| 5% of feature std | 0.848 | 0.857 |
| 10% of feature std | 0.865 | 0.847 |
| 25% of feature std | 0.794 | 0.827 |

The model degrades gracefully under noise rather than collapsing — even at 25% noise, precision drops only 6 percentage points from baseline. This suggests the model isn't relying on brittle, easily‑perturbed feature interactions.

**Fraud‑ratio resampling** (downsampling normal transactions while keeping all fraud cases, to test how metrics respond to a lower/higher fraud prevalence):

| Fraud Ratio | Precision | Recall |
|---|---|---|
| Baseline (0.17%) | 0.847 | 0.847 |
| 1% | 0.976 | 0.847 |
| 5% | 1.000 | 0.847 |
| 30% | 1.000 | 0.847 |

**Important caveat:** recall stays constant here by construction — since resampling only removes normal transactions and keeps every fraud case unchanged, the model sees the identical set of fraud transactions at every ratio. This demonstrates the base‑rate effect on precision (precision rises as normal transactions become relatively rarer), not evidence that the model generalizes to new or different fraud patterns. A genuine test of robustness to concept drift would require fraud examples the model hasn't seen, which this dataset's single time window doesn't provide.

### Sources informing this approach

- Chen et al. (2026), *A Regulatory Governance Framework for AI‑Driven Financial Fraud Detection in U.S. Banking*, [arXiv:2605.04076](https://arxiv.org/pdf/2605.04076) — net‑savings formula using empirical transaction value rather than assumed flat costs.
- FluxForce (2024), *False Positive Rates in Transaction Monitoring*, citing LexisNexis Risk Solutions' 2024 compliance cost survey of 1,000+ institutions — [source](https://www.fluxforce.ai/statistics/false-positive-rates-transaction-monitoring).
- EAI Endorsed Transactions (2026), *Enhancing Credit Card Fraud Detection under Severe Class Imbalance using Cost‑Sensitive Learning and Threshold Optimization* — a benchmark study using this same dataset, reducing false positives from 21 to 13 via threshold optimisation ([source](https://publications.eai.eu/index.php/ismla/article/view/12078)).
- Flagright (2024), *Understanding False Positives in Transaction Monitoring* — [source](https://www.flagright.com/post/understanding-false-positives-in-transaction-monitoring).
- INETCO / Corgi Labs (2019‑2026) — false decline cost research, cited above.

## Project Structure

```
fraud-detection-system/
├── data/                    # raw and processed datasets (not committed)
├── notebooks/               # exploratory data analysis
├── src/                     # core pipeline modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── risk_scoring.py
│   ├── cost_analysis.py
│   ├── expected_loss.py
│   ├── decision_layer.py
│   ├── system_simulation.py
│   ├── compare_models.py
│   └── interpretability.py
├── models/                  # saved model artifacts (not committed)
├── tests/                   # unit tests (29 tests)
└── app.py                   # optional demo interface
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
python -m src.train                # trains and saves all models
python -m src.evaluate              # prints precision/recall/F1/ROC‑AUC for all models
python -m src.predict               # runs a sample prediction
python -m src.risk_scoring          # shows risk score distribution by class
python -m src.cost_analysis         # prints the cost matrix (FN vs. FP)
python -m src.expected_loss         # finds the cost‑minimising threshold
python -m src.decision_layer        # builds the 3‑tier approve/review/block system
python -m src.system_simulation     # simulates full pipeline cost vs. no‑detection baseline, per model
python -m src.interpretability      # generates SHAP global importance and example explanations
```

6. **Run tests**

```bash
python -m pytest tests/ -v
```

## Challenges & Learnings

- **Class imbalance** was the core challenge — with fraud at just 0.17% of the data, accuracy became a meaningless metric. Switching evaluation focus to precision/recall/F1 was necessary to actually measure model usefulness.
- **Precision vs. recall trade‑off**: Logistic Regression and Random Forest optimise this trade‑off very differently. This project made it clear that "best model" depends on business priorities — is it worse to miss fraud, or to annoy customers with false flags?
- **Stratified splitting** was essential; without it, a random train/test split risks leaving too few (or zero) fraud examples in the test set, making evaluation unreliable.
- **Metrics alone don't make decisions**: precision and recall describe model behaviour, but they don't say what to *do* about it. Building the cost framework required grounding assumptions in real industry data rather than picking arbitrary cost values, since arbitrary costs would make any resulting "optimal threshold" meaningless.
- **A single threshold is still a blunt instrument**: even a cost‑optimal threshold forces a binary decision. Moving to a 3‑tier system (approve/review/block) cut manual review workload to 0.06% of transactions while keeping the same fraud‑catch rate — a meaningfully different (and more deployable) design than "pick one cutoff."
- **Not every cost is the same type of cost**: missed fraud, review labour, and false declines are economically different (lost money vs. labour cost vs. lost revenue/goodwill). Modelling them as three separate cost components, instead of collapsing everything into one FP/FN cost, produced a more honest picture — and surfaced a real gap (false‑block cost) that a simpler two‑cost model would have ignored entirely.
- **Classification metrics can be actively misleading in a cost‑sensitive system**: Logistic Regression's high recall looked like the strongest result on paper, but running the full cost simulation revealed it was worse than deploying nothing at all. Total system cost — not F1 or ROC‑AUC — is the metric that should drive model selection.
- **Interpretability matters for operational use**: SHAP explanations bridge the gap between black‑box predictions and the case‑level reasoning that fraud analysts and auditors require, especially when features are anonymised.

## Future Improvements

- Add a simple Streamlit demo (`app.py`) to interactively score transactions and see the tier assigned, along with its SHAP explanation
- Explore a more granular false‑block cost model incorporating customer lifetime value, following the false‑decline literature cited above
- Test true concept drift robustness using a dataset with multiple time periods, since this dataset's single 2‑day window limits how meaningfully fraud‑pattern drift can be simulated