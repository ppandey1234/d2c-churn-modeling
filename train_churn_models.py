import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_pipeline import (
    load_churn_labels,
    load_model_card,
    load_metrics,
    load_project_readme,
    load_rfm_snapshot,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("D2C_DATA_PATH", PROJECT_DIR / "data"))
PERFORMANCE_FILE = PROJECT_DIR / "model_performance.json"
FEATURE_IMPORTANCE_FILE = PROJECT_DIR / "feature_importances.json"
MODEL_SUMMARY_FILE = PROJECT_DIR / "model_summary.md"
MODEL_PIPELINE_FILE = PROJECT_DIR / "final_model_pipeline.joblib"


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"customer_id", "snapshot_date", "churn_next_60d", "split"}
    return [col for col in df.columns if col not in exclude]


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in feature_cols if pd.api.types.is_object_dtype(df[col])]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline(steps=[("model", model)])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.3) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5

    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    candidate = table[table["precision"] >= min_precision]
    if not candidate.empty:
        best = candidate.loc[candidate["recall"].idxmax()]
    else:
        best = table.loc[table["recall"].idxmax()]
    return float(best["threshold"])


def get_feature_names(preprocessor: ColumnTransformer, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in feature_cols if pd.api.types.is_object_dtype(df[col])]

    num_names = numeric_features
    cat_names = []
    if "cat" in preprocessor.named_transformers_:
        encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = encoder.get_feature_names_out(categorical_features).tolist()
    return num_names + cat_names


def build_model_pipelines(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Pipeline]:
    preprocessor = build_preprocessor(df, feature_cols)
    baseline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    final = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))])
    return {"baseline": baseline, "final": final}


def load_split_data() -> Dict[str, pd.DataFrame]:
    snapshot = load_rfm_snapshot()
    labels = load_churn_labels()
    snapshot = snapshot.drop(columns=["churn_next_60d", "split"], errors="ignore")
    merged = snapshot.merge(labels[["customer_id", "churn_next_60d", "split"]], on="customer_id", how="left")
    if merged["churn_next_60d"].isna().any():
        raise ValueError("Missing churn labels for some customers.")
    return {
        split: df.reset_index(drop=True)
        for split, df in merged.groupby("split")
    }


def print_documentation_summary() -> None:
    print("PROJECT CONTEXT")
    print("==============")
    print("Readme overview:\n")
    print(load_project_readme().split("---", 1)[0].strip())
    print("\nMODEL CARD SUMMARY:\n")
    model_card = load_model_card()
    for section in ["Model Approach", "Performance"]:
        if section in model_card:
            print(f"{section}:")
            print("\n".join(model_card[section]))
            print()


def train_and_evaluate() -> Dict[str, object]:
    data_splits = load_split_data()
    feature_cols = get_feature_columns(data_splits["train"])
    pipelines = build_model_pipelines(data_splits["train"], feature_cols)

    X_train = data_splits["train"][feature_cols]
    y_train = data_splits["train"]["churn_next_60d"].astype(int)
    X_valid = data_splits["validation"][feature_cols]
    y_valid = data_splits["validation"]["churn_next_60d"].astype(int)
    X_test = data_splits["test"][feature_cols]
    y_test = data_splits["test"]["churn_next_60d"].astype(int)

    results: Dict[str, object] = {"models": {}, "selected_threshold": None}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        prob_valid = pipeline.predict_proba(X_valid)[:, 1]
        threshold = select_threshold(y_valid, prob_valid, min_precision=0.3)

        valid_preds = (prob_valid >= threshold).astype(int)
        valid_metrics = evaluate_predictions(y_valid, valid_preds, prob_valid)

        prob_test = pipeline.predict_proba(X_test)[:, 1]
        test_preds = (prob_test >= threshold).astype(int)
        test_metrics = evaluate_predictions(y_test, test_preds, prob_test)

        model_result = {
            "threshold": threshold,
            "validation_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }

        if name == "final":
            model_result["feature_importances"] = extract_feature_importances(pipeline, data_splits["train"], feature_cols)
            save_model_pipeline(pipeline)
            results["selected_threshold"] = threshold

        results["models"][name] = model_result

    return results


def extract_feature_importances(pipeline: Pipeline, df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, float]]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocessor, df, feature_cols)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    return sorted(
        [
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(feature_names, importances)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )


def save_performance(results: Dict[str, object]) -> None:
    PERFORMANCE_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")


def save_feature_importances(results: Dict[str, object]) -> None:
    final_model = results["models"].get("final", {})
    importances = final_model.get("feature_importances")
    if importances is not None:
        FEATURE_IMPORTANCE_FILE.write_text(json.dumps(importances, indent=2), encoding="utf-8")


def save_model_pipeline(pipeline: Pipeline) -> None:
    joblib.dump(pipeline, MODEL_PIPELINE_FILE)


def generate_model_summary(results: Dict[str, object]) -> None:
    metrics = load_metrics()
    final = results["models"].get("final", {})
    lines = [
        "# Model Summary",
        "",
        "## Business and model context",
        "",
        "- Model name: Customer Churn Prediction Model",
        f"- Decision threshold preference: {metrics['decision_threshold']['business_preference']}",
        "- Higher recall was preferred due to the higher impact of missed churn-risk customers.",
        "",
        "## Final model performance",
        "",
    ]

    for split_name in ["validation", "test"]:
        split_metrics = final.get(f"{split_name}_metrics", {})
        lines.append(f"### {split_name.title()} Metrics")
        for metric_name, metric_value in split_metrics.items():
            lines.append(f"- {metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
        lines.append("")

    lines.extend(
        [
            "## Selected threshold",
            "",
            f"- Threshold used for final model: {results['selected_threshold']:.4f}",
            "",
            "## Important features",
            "",
        ]
    )
    for importance in final.get("feature_importances", [])[:15]:
        lines.append(f"- {importance['feature']}: {importance['importance']:.6f}")

    lines.extend(
        [
            "",
            "## Monitoring guidance",
            "",
            f"- Data drift: {metrics['monitoring_requirements']['data_drift']}",
            "- Prediction drift: " + ", ".join(metrics['monitoring_requirements']['prediction_drift']),
            "- Business outcomes: " + ", ".join(metrics['monitoring_requirements']['business_outcomes']),
            f"- Model performance: {metrics['monitoring_requirements']['model_performance']}",
        ]
    )
    MODEL_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print_documentation_summary()
    results = train_and_evaluate()
    save_performance(results)
    save_feature_importances(results)
    generate_model_summary(results)

    print("TRAINING COMPLETE")
    print(json.dumps(results, indent=2))
    print(f"Performance summary saved to {PERFORMANCE_FILE}")
    print(f"Feature importances saved to {FEATURE_IMPORTANCE_FILE}")
    print(f"Model summary saved to {MODEL_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
