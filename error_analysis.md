# Error Analysis Report

## Objective

Analyze incorrect predictions made by the churn prediction model.

The analysis focuses on:

- False Positives
- False Negatives

and their business impact.


---

# False Positives

False Positive:

Customer predicted as churn risk but actually stayed.


Business Risk:

- Unnecessary retention offers
- Increased marketing cost
- Customer annoyance


---

## Example 1

Customer ID: C1001

Prediction:

High churn risk


Actual:

Customer stayed


Reason:

Low activity was detected but customer had a temporary reduction in usage.


Action:

Avoid aggressive offers. Use low-cost engagement.


---

## Example 2

Customer ID: C1002

Prediction:

High churn risk


Actual:

Stayed


Reason:

High support requests were interpreted as dissatisfaction.


Action:

Review support quality before retention action.


---

## Example 3

Customer ID: C1003

Prediction:

High churn risk


Actual:

Stayed


Reason:

Short-term inactivity caused incorrect prediction.


Action:

Monitor future behavior.


---

# False Negatives

False Negative:

Customer predicted as safe but actually churned.


Business Risk:

- Lost customer
- Lost revenue
- Missed retention opportunity


---

## Example 4

Customer ID: C2001

Prediction:

Low churn risk


Actual:

Churned


Reason:

Model underestimated sudden decrease in usage.


Action:

Increase monitoring frequency.


---

## Example 5

Customer ID: C2002

Prediction:

Low risk


Actual:

Churned


Reason:

Recent complaints were not weighted strongly enough.


Action:

Include stronger customer experience features.


---

## Example 6

Customer ID: C2003

Prediction:

Low risk


Actual:

Churned


Reason:

Payment issues appeared before churn.


Action:

Create payment-risk alerts.


---

# Additional Customer Cases


| Customer | Error Type | Reason | Business Impact |
|---|---|---|---|
| C3001 | FP | Temporary inactivity | Extra campaign cost |
| C3002 | FP | Seasonal usage drop | Unnecessary contact |
| C3003 | FP | Support spike | Customer irritation |
| C4001 | FN | Usage decline missed | Customer loss |
| C4002 | FN | Complaint ignored | Revenue loss |
| C4003 | FN | Payment issue missed | Churn risk |


---

# Summary

False negatives are considered more costly because they represent customers who leave without intervention.

The selected threshold balances recall and operational cost.
