"""Matplotlib visualizations for EDA, evaluation, and explainability."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


def ensure_figure_dir(output_dir: Path) -> Path:
    """Create and return outputs/figures."""
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def _save_figure(path: Path) -> Path:
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_target_distribution(df, output_dir: Path) -> Path:
    counts = df["default_next_month"].value_counts().sort_index()
    plt.figure(figsize=(7, 4.5))
    plt.bar(["No Default", "Default"], counts.values, color=["#4C78A8", "#E45756"])
    plt.title("Next-Month Default Distribution")
    plt.xlabel("Default Status")
    plt.ylabel("Number of Clients")
    return _save_figure(ensure_figure_dir(output_dir) / "target_distribution.png")


def plot_default_rate_by_repay_status(df, output_dir: Path) -> Path:
    rates = df.groupby("repay_status_sep")["default_next_month"].mean().sort_index()
    plt.figure(figsize=(8, 4.5))
    plt.bar(rates.index.astype(str), rates.values, color="#F2CF5B")
    plt.title("Default Rate by September Repayment Status")
    plt.xlabel("September Repayment Status")
    plt.ylabel("Actual Default Rate")
    plt.ylim(0, min(1.0, max(rates.max() * 1.15, 0.1)))
    return _save_figure(ensure_figure_dir(output_dir) / "default_rate_by_repay_status_sep.png")


def plot_credit_limit_distribution(df, output_dir: Path) -> Path:
    plt.figure(figsize=(8, 4.5))
    bins = np.linspace(df["credit_limit"].min(), df["credit_limit"].max(), 35)
    plt.hist(
        df.loc[df["default_next_month"] == 0, "credit_limit"],
        bins=bins,
        alpha=0.60,
        density=True,
        label="No Default",
        color="#4C78A8",
    )
    plt.hist(
        df.loc[df["default_next_month"] == 1, "credit_limit"],
        bins=bins,
        alpha=0.60,
        density=True,
        label="Default",
        color="#E45756",
    )
    plt.title("Credit Limit Distribution by Default Status")
    plt.xlabel("Credit Limit")
    plt.ylabel("Density")
    plt.legend()
    return _save_figure(ensure_figure_dir(output_dir) / "credit_limit_distribution.png")


def plot_roc_curves(models, X_test, y_test, output_dir: Path) -> Path:
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    return _save_figure(ensure_figure_dir(output_dir) / "roc_curves.png")


def plot_precision_recall_curves(models, X_test, y_test, output_dir: Path) -> Path:
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for name, model in models.items():
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.axhline(float(np.mean(y_test)), linestyle="--", color="gray", label="Portfolio default rate")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    return _save_figure(ensure_figure_dir(output_dir) / "precision_recall_curves.png")


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_dir: Path) -> Path:
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=["No Default", "Default"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {model_name}")
    return _save_figure(
        ensure_figure_dir(output_dir) / f"confusion_matrix_{_safe_name(model_name)}.png"
    )


def plot_score_distribution(y_true, y_prob, model_name: str, output_dir: Path) -> Path:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    plt.figure(figsize=(8, 4.5))
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.65, density=True, label="No Default", color="#4C78A8")
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.65, density=True, label="Default", color="#E45756")
    plt.title(f"Predicted Default Probability: {model_name}")
    plt.xlabel("Predicted Default Probability")
    plt.ylabel("Density")
    plt.legend()
    return _save_figure(
        ensure_figure_dir(output_dir) / f"score_distribution_{_safe_name(model_name)}.png"
    )


def plot_feature_importance(model, output_dir: Path, top_n: int = 15) -> Path:
    """Plot native tree importance or absolute logistic-regression coefficients."""
    estimator = model.named_steps.get("model", model)
    preprocessor = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None

    importance = None
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_)).ravel()

    figure_path = ensure_figure_dir(output_dir) / "feature_importance.png"
    if importance is None:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, "Native feature importance is not available for this model.", ha="center", va="center")
        plt.title("Feature Importance")
        return _save_figure(figure_path)

    try:
        feature_names = np.asarray(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = np.asarray([f"feature_{index}" for index in range(len(importance))])
    if len(feature_names) != len(importance):
        feature_names = np.asarray([f"feature_{index}" for index in range(len(importance))])

    top_indices = np.argsort(importance)[-top_n:]
    clean_names = [name.split("__", 1)[-1] for name in feature_names[top_indices]]
    plt.figure(figsize=(9, 6))
    plt.barh(clean_names, importance[top_indices], color="#59A14F")
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    return _save_figure(figure_path)


def plot_risk_tier_summary(summary_df, output_dir: Path) -> Path:
    plt.figure(figsize=(8, 4.5))
    plt.bar(summary_df["risk_tier"].astype(str), summary_df["actual_default_rate"], color="#B279A2")
    plt.title("Actual Default Rate by Risk Tier")
    plt.xlabel("Risk Tier")
    plt.ylabel("Actual Default Rate")
    plt.ylim(0, min(1.0, max(summary_df["actual_default_rate"].max() * 1.15, 0.1)))
    return _save_figure(ensure_figure_dir(output_dir) / "risk_tier_default_rate.png")


def plot_lift_curve(decile_df, output_dir: Path) -> Path:
    plt.figure(figsize=(7, 5))
    x = decile_df["cumulative_population_share"]
    y = decile_df["cumulative_default_capture_rate"]
    plt.plot(x, y, marker="o", linewidth=2, color="#E15759", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random selection")
    plt.title("Cumulative Gains Curve")
    plt.xlabel("Cumulative Population Reviewed")
    plt.ylabel("Cumulative Defaults Captured")
    plt.legend()
    plt.grid(alpha=0.25)
    return _save_figure(ensure_figure_dir(output_dir) / "cumulative_gains_curve.png")


def plot_decile_default_rate(decile_df, output_dir: Path) -> Path:
    plt.figure(figsize=(8, 4.5))
    plt.bar(decile_df["risk_decile"].astype(str), decile_df["default_rate"], color="#F28E2B")
    plt.axhline(
        np.average(decile_df["default_rate"], weights=decile_df["n_clients"]),
        linestyle="--",
        color="gray",
        label="Portfolio average",
    )
    plt.title("Actual Default Rate by Risk Decile")
    plt.xlabel("Risk Decile (1 = Highest Risk)")
    plt.ylabel("Actual Default Rate")
    plt.legend()
    return _save_figure(ensure_figure_dir(output_dir) / "decile_default_rate.png")


def plot_threshold_tradeoff(threshold_df, selected_threshold: float, output_dir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    plt.axvline(selected_threshold, linestyle="--", color="#E15759", label=f"Selected: {selected_threshold:.3f}")
    plt.title("Classification Threshold Trade-off")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(alpha=0.25)
    return _save_figure(ensure_figure_dir(output_dir) / "threshold_tradeoff.png")


def plot_threshold_cost(threshold_df, selected_threshold: float, output_dir: Path) -> Path:
    plt.figure(figsize=(8, 4.5))
    plt.plot(threshold_df["threshold"], threshold_df["cost_per_client"], color="#B07AA1")
    plt.axvline(selected_threshold, linestyle="--", color="#E15759", label=f"Selected: {selected_threshold:.3f}")
    plt.title("Business Cost by Classification Threshold")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Assumed Cost per Client")
    plt.legend()
    plt.grid(alpha=0.25)
    return _save_figure(ensure_figure_dir(output_dir) / "threshold_business_cost.png")


def plot_calibration_curve(y_true, y_prob, model_name: str, output_dir: Path) -> Path:
    observed, predicted = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(predicted, observed, marker="o", linewidth=2, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.title("Probability Calibration Curve")
    plt.xlabel("Mean Predicted Default Probability")
    plt.ylabel("Observed Default Rate")
    plt.legend()
    plt.grid(alpha=0.25)
    return _save_figure(ensure_figure_dir(output_dir) / "calibration_curve.png")


def plot_permutation_importance(importance_df, output_dir: Path, top_n: int = 20) -> Path:
    top = importance_df.head(top_n).sort_values("importance_mean")
    plt.figure(figsize=(9, 7))
    plt.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#59A14F",
        alpha=0.85,
    )
    plt.title("Permutation Feature Importance")
    plt.xlabel("Mean Decrease in ROC-AUC")
    plt.ylabel("Feature")
    return _save_figure(ensure_figure_dir(output_dir) / "permutation_feature_importance.png")


def plot_cv_stability(cv_df, output_dir: Path) -> Path:
    ordered = cv_df.sort_values("roc_auc_mean")
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        ordered["roc_auc_mean"],
        ordered["model"],
        xerr=ordered["roc_auc_std"],
        fmt="o",
        capsize=4,
        color="#4E79A7",
    )
    plt.title("Cross-Validated ROC-AUC Stability")
    plt.xlabel("Mean ROC-AUC with Standard Deviation")
    plt.ylabel("Model")
    plt.grid(axis="x", alpha=0.25)
    return _save_figure(ensure_figure_dir(output_dir) / "cross_validation_stability.png")
