import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("D2C_DATA_PATH", ROOT_DIR / "data"))
README_FILE = ROOT_DIR / "README.md"
MODEL_CARD_FILE = ROOT_DIR / "model_card.md"
METRICS_FILE = ROOT_DIR / "monitoring.json"


def load_rfm_snapshot() -> pd.DataFrame:
    """Load the pre-built modeling snapshot from the data folder."""
    path = DATA_DIR / "rfm_modeling_snapshot.csv"
    return pd.read_csv(path)


def load_churn_labels() -> pd.DataFrame:
    """Load churn target labels and train/validation/test split."""
    path = DATA_DIR / "churn_labels.csv"
    return pd.read_csv(path)


def load_project_readme() -> str:
    """Load the README overview to keep the code aligned with project context."""
    return README_FILE.read_text(encoding="utf-8")


def load_model_card() -> Dict[str, List[str]]:
    """Parse the model card markdown into simple section text."""
    content = MODEL_CARD_FILE.read_text(encoding="utf-8")
    sections: Dict[str, List[str]] = {}
    current_title = "Overview"
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("## ") or line.startswith("# "):
            current_title = line.lstrip("# ").strip()
            sections[current_title] = []
        elif current_title:
            if line:
                sections.setdefault(current_title, []).append(line)
    return sections


def load_metrics() -> dict:
    """Load model monitoring and ethical risk details from monitoring.json."""
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


def merge_labels(snapshot: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join the snapshot features with churn labels for analysis."""
    merged = snapshot.merge(labels[["customer_id", "churn_next_60d", "split"]], on="customer_id", how="left")
    if merged["churn_next_60d"].isna().any():
        raise ValueError("Missing churn labels for some customers in the snapshot.")
    return merged


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    """Produce a small summary report for the dataset."""
    summary = {
        "shape": df.shape,
        "feature_count": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "churn_rate": float(df["churn_next_60d"].mean()),
        "split_counts": df["split"].value_counts(dropna=False).to_dict(),
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
    }

    categorical_columns = [col for col in df.columns if df[col].dtype == object and col not in ["customer_id", "snapshot_date"]]
    summary["categorical_unique_counts"] = {
        col: int(df[col].nunique(dropna=False)) for col in categorical_columns
    }
    return summary


def print_project_summary() -> None:
    s = load_project_readme().split("---", 1)[0].strip()
    print("PROJECT OVERVIEW:\n")
    print(s)
    print("\n---\n")

    model_card = load_model_card()
    print("MODEL CARD SECTIONS:")
    for section in ["Model Name", "Intended Use", "Model Approach", "Performance"]:
        if section in model_card:
            print(f"\n{section}:")
            print("  ".join(model_card[section]))

    metrics = load_metrics()
    print("\n---\n")
    print("KEY METRICS & MONITORING:")
    print(f"Decision preference: {metrics['decision_threshold']['business_preference']}")
    print(f"Monitoring focus: {', '.join(metrics['monitoring_requirements']['prediction_drift'])}")


def main() -> None:
    snapshot = load_rfm_snapshot()
    labels = load_churn_labels()
    merged = merge_labels(snapshot, labels)
    report = summarize_dataframe(merged)

    print_project_summary()
    print("\nDATA SUMMARY:\n")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nTop 5 rows of merged data:")
    print(merged.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
