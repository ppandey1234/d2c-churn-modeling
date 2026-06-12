import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("D2C_DATA_PATH", PROJECT_DIR / "data"))
MODEL_FILE = PROJECT_DIR / "final_model_pipeline.joblib"
PERFORMANCE_FILE = PROJECT_DIR / "model_performance.json"
INPUT_FILE = DATA_DIR / "rfm_modeling_snapshot.csv"
OUTPUT_FILE = PROJECT_DIR / "predictions.csv"


def load_model() -> object:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}. Run train_churn_models.py first.")
    return joblib.load(MODEL_FILE)


def load_threshold() -> float:
    if not PERFORMANCE_FILE.exists():
        raise FileNotFoundError(f"Performance file not found: {PERFORMANCE_FILE}. Run train_churn_models.py first.")
    data = json.loads(PERFORMANCE_FILE.read_text(encoding="utf-8"))
    return float(data.get("selected_threshold", 0.5))


def load_input_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    if "churn_next_60d" in df.columns:
        df = df.drop(columns=["churn_next_60d"], errors="ignore")
    if "split" in df.columns:
        df = df.drop(columns=["split"], errors="ignore")
    return df


def predict(model: object, threshold: float, df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col not in ["customer_id", "snapshot_date"]]
    X = df[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    output = df[["customer_id"]].copy()
    output["churn_probability"] = probabilities
    output["predicted_churn"] = predictions
    output["threshold"] = threshold
    if "split" in df.columns:
        output["split"] = df["split"]
    return output


def save_predictions(predictions: pd.DataFrame, path: Path) -> None:
    predictions.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate churn predictions using the trained pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Input feature snapshot CSV file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_FILE,
        help="Saved pipeline file.",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=PERFORMANCE_FILE,
        help="Model performance JSON containing the selected threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output CSV file for predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    threshold = float(json.loads(args.performance.read_text(encoding="utf-8")).get("selected_threshold", 0.5))
    df = pd.read_csv(args.input)
    feature_df = df.drop(columns=[col for col in ["churn_next_60d", "split"] if col in df.columns], errors="ignore")
    feature_cols = [col for col in feature_df.columns if col not in ["customer_id", "snapshot_date"]]
    probabilities = model.predict_proba(feature_df[feature_cols])[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    output = pd.DataFrame(
        {
            "customer_id": feature_df["customer_id"],
            "churn_probability": probabilities,
            "predicted_churn": predictions,
            "threshold": threshold,
        }
    )
    if "split" in df.columns:
        output["split"] = df["split"]

    output.to_csv(args.output, index=False)
    print(f"Saved predictions for {len(output)} customers to {args.output}")


if __name__ == "__main__":
    main()
