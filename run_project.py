"""Run the complete, research-grade credit default risk modeling workflow."""

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd

from src.data_preprocessing import (
    build_preprocessor,
    clean_credit_data,
    load_credit_default_data,
    split_features_target,
)
from src.evaluation import (
    create_risk_tiers,
    decile_lift_summary,
    evaluate_models,
    evaluate_probabilities,
    permutation_feature_importance,
    risk_tier_summary,
    save_metrics,
    save_risk_tier_summary,
    save_table,
    select_cost_optimal_threshold,
    threshold_analysis,
)
from src.model_training import (
    calibrate_model,
    compare_feature_engineering_cv,
    cross_validate_models,
    get_oof_probabilities,
    split_train_test,
    train_models,
    tune_hist_gradient_boosting,
)
from src.visualization import (
    ensure_figure_dir,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_credit_limit_distribution,
    plot_cv_stability,
    plot_decile_default_rate,
    plot_default_rate_by_repay_status,
    plot_feature_importance,
    plot_lift_curve,
    plot_permutation_importance,
    plot_precision_recall_curves,
    plot_risk_tier_summary,
    plot_roc_curves,
    plot_score_distribution,
    plot_target_distribution,
    plot_threshold_cost,
    plot_threshold_tradeoff,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run credit default risk modeling.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer tuning iterations and CV folds for a quicker smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    tuning_iterations = 4 if args.fast else 12
    stability_cv_folds = 3 if args.fast else 5
    permutation_repeats = 3 if args.fast else 5

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "outputs"
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    ensure_figure_dir(output_dir)

    print("1/8 Loading, cleaning, and validating data...")
    df = clean_credit_data(load_credit_default_data(data_dir))
    plot_target_distribution(df, output_dir)
    plot_default_rate_by_repay_status(df, output_dir)
    plot_credit_limit_distribution(df, output_dir)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("2/8 Training benchmark models with leakage-safe business feature engineering...")
    preprocessor = build_preprocessor(include_engineered=True)
    models = train_models(X_train, y_train, preprocessor)

    print("3/8 Tuning HistGradientBoosting using training-only cross-validation...")
    tuned_model, search_results = tune_hist_gradient_boosting(
        X_train,
        y_train,
        preprocessor,
        n_iter=tuning_iterations,
        cv_folds=3,
    )
    models["Tuned HistGradientBoosting"] = tuned_model
    save_table(search_results, output_dir, "hyperparameter_search_results.csv")

    print("4/8 Measuring cross-validation stability and feature-engineering contribution...")
    cv_metrics = cross_validate_models(models, X_train, y_train, cv_folds=stability_cv_folds)
    feature_impact = compare_feature_engineering_cv(X_train, y_train, cv_folds=stability_cv_folds)
    save_table(cv_metrics, output_dir, "cross_validation_metrics.csv")
    save_table(feature_impact, output_dir, "feature_engineering_impact.csv")
    plot_cv_stability(cv_metrics, output_dir)

    # Model selection uses training-set CV only. The test set remains a final unbiased holdout.
    best_model_name = str(cv_metrics.iloc[0]["model"])
    best_ranking_model = models[best_model_name]

    print("5/8 Calibrating the selected model and selecting a business threshold...")
    calibrated_model = calibrate_model(best_ranking_model, X_train, y_train, cv_folds=3)
    calibrated_model_name = f"Calibrated {best_model_name}"
    reporting_models = {**models, calibrated_model_name: calibrated_model}

    raw_oof_prob = get_oof_probabilities(best_ranking_model, X_train, y_train, cv_folds=3)
    calibrated_oof_prob = get_oof_probabilities(calibrated_model, X_train, y_train, cv_folds=3)
    calibration_comparison = pd.DataFrame(
        [
            evaluate_probabilities(best_model_name, y_train, raw_oof_prob),
            evaluate_probabilities(calibrated_model_name, y_train, calibrated_oof_prob),
        ]
    ).sort_values(["brier_score", "log_loss"])
    save_table(calibration_comparison, output_dir, "calibration_comparison.csv")

    operational_model_name = str(calibration_comparison.iloc[0]["model"])
    if operational_model_name == calibrated_model_name:
        operational_model = calibrated_model
        oof_prob = calibrated_oof_prob
    else:
        operational_model = best_ranking_model
        oof_prob = raw_oof_prob

    threshold_df = threshold_analysis(
        y_train,
        oof_prob,
        false_negative_cost=5.0,
        false_positive_cost=1.0,
    )
    selected_threshold = select_cost_optimal_threshold(threshold_df)
    save_table(threshold_df, output_dir, "threshold_analysis.csv")
    plot_threshold_tradeoff(threshold_df, selected_threshold, output_dir)
    plot_threshold_cost(threshold_df, selected_threshold, output_dir)

    print("6/8 Evaluating the untouched test set and building portfolio reports...")
    metrics_df = evaluate_models(reporting_models, X_test, y_test)
    save_metrics(metrics_df, output_dir)

    y_prob = operational_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= selected_threshold).astype(int)
    threshold_metrics = pd.DataFrame(
        [
            evaluate_probabilities(operational_model_name + " | threshold=0.50", y_test, y_prob, 0.50),
            evaluate_probabilities(
                operational_model_name + " | cost-optimised threshold",
                y_test,
                y_prob,
                selected_threshold,
            ),
        ]
    )
    save_table(threshold_metrics, output_dir, "threshold_metrics.csv")

    summary_df = risk_tier_summary(y_test, y_prob)
    decile_df = decile_lift_summary(y_test, y_prob)
    save_risk_tier_summary(summary_df, output_dir)
    save_table(decile_df, output_dir, "decile_lift_summary.csv")

    predictions = pd.DataFrame(
        {
            "actual_default": y_test.to_numpy(),
            "predicted_default_probability": y_prob,
            "risk_tier": create_risk_tiers(y_prob).astype(str),
            "predicted_default_at_selected_threshold": y_pred,
        }
    )
    save_table(predictions, output_dir, "test_predictions.csv")

    print("7/8 Producing explainability, ranking, calibration, and decision-policy charts...")
    importance_df = permutation_feature_importance(
        best_ranking_model,
        X_test,
        y_test,
        n_repeats=permutation_repeats,
    )
    save_table(importance_df, output_dir, "permutation_feature_importance.csv")

    plot_roc_curves(reporting_models, X_test, y_test, output_dir)
    plot_precision_recall_curves(reporting_models, X_test, y_test, output_dir)
    plot_confusion_matrix(y_test, y_pred, operational_model_name, output_dir)
    plot_score_distribution(y_test, y_prob, operational_model_name, output_dir)
    plot_calibration_curve(y_test, y_prob, operational_model_name, output_dir)
    plot_risk_tier_summary(summary_df, output_dir)
    plot_lift_curve(decile_df, output_dir)
    plot_decile_default_rate(decile_df, output_dir)
    plot_permutation_importance(importance_df, output_dir)
    plot_feature_importance(models["Random Forest"], output_dir)

    print("8/8 Saving the operational model and reproducibility metadata...")
    joblib.dump(operational_model, model_dir / "best_operational_model.joblib")
    metadata = pd.DataFrame(
        [
            {
                "selected_by": f"{stability_cv_folds}-fold training CV ROC-AUC",
                "best_ranking_model": best_model_name,
                "operational_model": operational_model_name,
                "selected_threshold": selected_threshold,
                "false_negative_cost": 5.0,
                "false_positive_cost": 1.0,
                "training_rows": len(X_train),
                "test_rows": len(X_test),
                "random_state": 42,
                "fast_mode": args.fast,
            }
        ]
    )
    save_table(metadata, output_dir, "model_metadata.csv")

    print("\nFinal holdout model metrics:")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nCV-selected ranking model: {best_model_name}")
    print(f"Operational probability model: {operational_model_name}")
    print(f"Training-OOF cost-optimised threshold: {selected_threshold:.3f}")
    print("\nThreshold comparison:")
    print(
        threshold_metrics[["model", "threshold", "precision", "recall", "f1", "accuracy"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nRisk tier summary:")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n项目优化与验证已完成，可查看 outputs/ 中的高级评估报告、图表和模型文件。")


if __name__ == "__main__":
    main()
