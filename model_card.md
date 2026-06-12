# Model Card

## Model Name

Customer Churn Prediction Model

---

# Intended Use

This model predicts customers who are likely to churn within the next 60 days.

The output is intended to help retention teams:

- prioritize customers
- design engagement strategies
- allocate retention resources

---

# Users

Primary users:

- Customer success teams
- Marketing teams
- Business analysts

---

# Data Used

The model uses customer information available before the prediction date.

Features may include:

- customer activity
- usage patterns
- support interactions
- subscription details
- payment history


Future churn-related information is excluded.

---

# Model Approach

Two models were developed:

## Baseline

Logistic Regression


## Final Model

Random Forest Classifier


Random Forest was selected because it captures complex relationships and provides feature importance.

---

# Performance

Metrics reported:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC


Complete values are available in:
