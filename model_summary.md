# Model Summary

## Business and model context

- Model name: Customer Churn Prediction Model
- Decision threshold preference: Higher recall was preferred because failing to identify a churn-risk customer has higher impact.
- Higher recall was preferred due to the higher impact of missed churn-risk customers.

## Final model performance

### Validation Metrics
- Accuracy: 0.4375
- Precision: 0.4375
- Recall: 1.0000
- F1 Score: 0.6087
- Roc Auc: 0.8672

### Test Metrics
- Accuracy: 0.5000
- Precision: 0.5000
- Recall: 1.0000
- F1 Score: 0.6667
- Roc Auc: 0.8809

## Selected threshold

- Threshold used for final model: 0.0150

## Important features

- recency_days: 0.188668
- last_visit_days_ago: 0.147037
- monetary_180d: 0.081752
- days_since_signup: 0.054428
- product_views_30d: 0.053875
- avg_discount_pct_180d: 0.047289
- frequency_180d: 0.039528
- sessions_30d: 0.037125
- category_diversity_180d: 0.028575
- avg_rating_180d: 0.026886
- email_opens_30d: 0.024360
- cart_adds_30d: 0.020892
- wishlist_adds_30d: 0.018455
- avg_resolution_hours_90d: 0.016153
- campaign_clicks_30d: 0.014474

## Monitoring guidance

- Data drift: Check whether customer behavior changes.
- Prediction drift: churn probability, high-risk customer percentage
- Business outcomes: actual churn rate, retention success
- Model performance: Retrain when performance decreases.