# Customer Churn Prediction Model

## Project Overview

This project builds a machine learning model to identify customers who are likely to churn within the next 60 days.

The objective is to support customer retention teams by identifying high-risk customers early and enabling targeted retention actions.

The project includes:

- Data preparation
- Leakage prevention checks
- Baseline model
- Advanced machine learning model
- Model evaluation
- Threshold selection
- Feature importance analysis
- Error analysis
- Model documentation


---

# Dataset

The model uses the provided customer churn dataset.

Only information available on or before the customer snapshot date is used.

Future information related to churn outcome or post-snapshot behavior was excluded to prevent data leakage.


---

# Project Structure

## Usage

Run the data pipeline and model training from the project root:

1. Inspect data and project context:
   ```bash
   python data_pipeline.py
   ```
2. Train baseline and final models, then save performance and feature summaries:
   ```bash
   python train_churn_models.py
   ```
3. Generate churn predictions using the trained final model:
   ```bash
   python predict_churn.py
   ```

If your dataset lives outside the default `data/` folder, set the environment variable first:

```bash
set D2C_DATA_PATH=E:\path\to\data
python train_churn_models.py
python predict_churn.py
```

## Output files

- `model_performance.json` - final and baseline evaluation metrics, selected threshold
- `metrics.json` - final evaluation metrics for the selected model
- `feature_importances.json` - final model feature importance ranking
- `model_summary.md` - narrative summary of final model performance and monitoring guidance
- `monitoring.json` - monitoring and ethical risk metadata
- `final_model_pipeline.joblib` - serialized final Random Forest pipeline
- `predictions.csv` - predicted churn probabilities and labels for snapshot customers

