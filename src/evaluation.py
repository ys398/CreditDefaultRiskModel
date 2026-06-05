"""Model evaluation, threshold analysis, and portfolio reporting utilities."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


RISK_TIER_ORDER = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]


def ks_statistic(y_true, y_prob) -> float:
    """Calculate the maximum separation between cumulative bad and good rates."""
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    frame = frame.sort_values("y_prob", ascending=False)
    positives = (frame["y_true"] == 1).sum()
    negatives = (frame["y_true"] == 0).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    cumulative_positive = (frame["y_true"] == 1).cumsum() / positives
    cumulative_negative = (frame["y_true"] == 0).cumsum() / negatives
    return float((cumulative_positive - cumulative_negative).abs().max())


def capture_rate_at_fraction(y_true, y_prob, fraction: float) -> float:
    """Return the share of all defaults captured in the riskiest portfolio fraction."""
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    n_selected = max(1, int(np.ceil(len(frame) * fraction)))
    selected_defaults = frame.nlargest(n_selected, "y_prob")["y_true"].sum()
    total_defaults = frame["y_true"].sum()
    return float(selected_defaults / total_defaults) if total_defaults else float("nan")


def lift_at_fraction(y_true, y_prob, fraction: float) -> float:
    """Return default-rate lift in the riskiest portfolio fraction."""
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    n_selected = max(1, int(np.ceil(len(frame) * fraction)))
    portfolio_rate = frame["y_true"].mean()
    selected_rate = frame.nlargest(n_selected, "y_prob")["y_true"].mean()
    return float(selected_rate / portfolio_rate) if portfolio_rate else float("nan")


def evaluate_probabilities(name: str, y_true, y_prob, threshold: float = 0.5) -> Dict:
    """Evaluate probability ranking, calibration, and threshold classification."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "ks": ks_statistic(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "top_10_capture_rate": capture_rate_at_fraction(y_true, y_prob, 0.10),
        "top_20_capture_rate": capture_rate_at_fraction(y_true, y_prob, 0.20),
        "top_10_lift": lift_at_fraction(y_true, y_prob, 0.10),
        "top_20_lift": lift_at_fraction(y_true, y_prob, 0.20),
    }


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict:
    """Evaluate one fitted model using ranking and classification metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_probabilities(name, y_test, y_prob, threshold=threshold)


def evaluate_models(models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate and rank all fitted models."""
    rows = [evaluate_model(name, model, X_test, y_test) for name, model in models.items()]
    return pd.DataFrame(rows).sort_values(["roc_auc", "ks"], ascending=False).reset_index(drop=True)


def threshold_analysis(
    y_true,
    y_prob,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> pd.DataFrame:
    """Evaluate operating thresholds under an explicit asymmetric cost assumption."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []
    for threshold in np.linspace(0.05, 0.80, 151):
        y_pred = (y_prob >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        total_cost = false_negative_cost * fn + false_positive_cost * fp
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "specificity": tn / (tn + fp) if tn + fp else float("nan"),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "predicted_positive_rate": y_pred.mean(),
                "false_positives": fp,
                "false_negatives": fn,
                "total_cost": total_cost,
                "cost_per_client": total_cost / len(y_true),
            }
        )
    return pd.DataFrame(rows)


def select_cost_optimal_threshold(threshold_df: pd.DataFrame) -> float:
    """Select the minimum-cost threshold, preferring higher recall on ties."""
    ranked = threshold_df.sort_values(["total_cost", "recall"], ascending=[True, False])
    return float(ranked.iloc[0]["threshold"])


def create_risk_tiers(y_prob):
    """Convert predicted probabilities into ordered percentile-based risk tiers."""
    probabilities = pd.Series(np.asarray(y_prob), name="predicted_default_prob")
    percentile_rank = probabilities.rank(method="first", pct=True)
    tiers = pd.cut(
        percentile_rank,
        bins=[0.0, 0.50, 0.80, 0.95, 1.0],
        labels=RISK_TIER_ORDER,
        include_lowest=True,
        ordered=True,
    )
    return tiers


def risk_tier_summary(y_true, y_prob) -> pd.DataFrame:
    """Summarize portfolio size and observed/predicted risk by tier."""
    frame = pd.DataFrame(
        {
            "actual_default": np.asarray(y_true),
            "predicted_default_prob": np.asarray(y_prob),
            "risk_tier": create_risk_tiers(y_prob),
        }
    )
    summary = (
        frame.groupby("risk_tier", observed=False)
        .agg(
            n_clients=("actual_default", "size"),
            actual_default_rate=("actual_default", "mean"),
            avg_predicted_default_prob=("predicted_default_prob", "mean"),
            min_predicted_default_prob=("predicted_default_prob", "min"),
            max_predicted_default_prob=("predicted_default_prob", "max"),
        )
        .reset_index()
    )
    summary.insert(2, "share", summary["n_clients"] / len(frame))
    summary["risk_tier"] = pd.Categorical(summary["risk_tier"], categories=RISK_TIER_ORDER, ordered=True)
    return summary.sort_values("risk_tier").reset_index(drop=True)


def decile_lift_summary(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Create decile-level lift and cumulative default-capture reporting."""
    frame = pd.DataFrame({"actual_default": np.asarray(y_true), "predicted_default_prob": np.asarray(y_prob)})
    frame = frame.sort_values("predicted_default_prob", ascending=False).reset_index(drop=True)
    frame["risk_decile"] = pd.qcut(
        frame.index,
        q=n_bins,
        labels=range(1, n_bins + 1),
    ).astype(int)
    portfolio_default_rate = frame["actual_default"].mean()
    total_defaults = frame["actual_default"].sum()
    summary = (
        frame.groupby("risk_decile")
        .agg(
            n_clients=("actual_default", "size"),
            defaults=("actual_default", "sum"),
            default_rate=("actual_default", "mean"),
            avg_predicted_default_prob=("predicted_default_prob", "mean"),
        )
        .reset_index()
    )
    summary["population_share"] = summary["n_clients"] / len(frame)
    summary["cumulative_population_share"] = summary["population_share"].cumsum()
    summary["default_capture_rate"] = summary["defaults"] / total_defaults
    summary["cumulative_default_capture_rate"] = summary["default_capture_rate"].cumsum()
    summary["lift"] = summary["default_rate"] / portfolio_default_rate
    summary["cumulative_lift"] = (
        summary["cumulative_default_capture_rate"] / summary["cumulative_population_share"]
    )
    return summary


def permutation_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    """Calculate transformed-feature permutation importance for a fitted pipeline."""
    if not hasattr(model, "named_steps"):
        raise ValueError("Permutation feature importance requires a fitted sklearn Pipeline.")

    transformed_X = X_test
    if "feature_engineering" in model.named_steps:
        transformed_X = model.named_steps["feature_engineering"].transform(transformed_X)
    preprocessor = model.named_steps["preprocessor"]
    transformed_X = preprocessor.transform(transformed_X)
    estimator = model.named_steps["model"]

    result = permutation_importance(
        estimator,
        transformed_X,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{index}" for index in range(transformed_X.shape[1])]

    importance_df = pd.DataFrame(
        {
            "feature": [str(name).split("__", 1)[-1] for name in feature_names],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def save_table(df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    """Save a dataframe under outputs/ using a supplied filename."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path


def save_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    """Save model metrics to outputs/model_metrics.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_metrics.csv"
    metrics_df.to_csv(path, index=False)
    return path


def save_risk_tier_summary(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    """Save risk-tier reporting to outputs/risk_tier_summary.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "risk_tier_summary.csv"
    summary_df.to_csv(path, index=False)
    return path
