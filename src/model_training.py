"""Model training, tuning, calibration, and validation utilities."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor
from src.feature_engineering import CreditRiskFeatureEngineer


def split_train_test(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """Create a stratified 80/20 train-test split."""
    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )


def build_model_pipeline(estimator, preprocessor=None, use_feature_engineering: bool = True) -> Pipeline:
    """Build a leakage-safe modeling pipeline."""
    if preprocessor is None:
        preprocessor = build_preprocessor(include_engineered=use_feature_engineering)
    steps = []
    if use_feature_engineering:
        steps.append(("feature_engineering", CreditRiskFeatureEngineer()))
    steps.extend([("preprocessor", clone(preprocessor)), ("model", estimator)])
    return Pipeline(steps)


def train_models(X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> Dict[str, Pipeline]:
    """Train three benchmark credit-risk models using leakage-safe pipelines."""
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    }

    models = {}
    for name, estimator in estimators.items():
        pipeline = build_model_pipeline(estimator, preprocessor, use_feature_engineering=True)
        pipeline.fit(X_train, y_train)
        models[name] = pipeline
    return models


def tune_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    n_iter: int = 12,
    cv_folds: int = 3,
) -> Tuple[Pipeline, pd.DataFrame]:
    """Tune HistGradientBoosting using training-only stratified CV and ROC-AUC."""
    pipeline = build_model_pipeline(
        HistGradientBoostingClassifier(random_state=42),
        preprocessor,
        use_feature_engineering=True,
    )
    param_distributions = {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
        "model__max_iter": [150, 250, 350],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [20, 50, 100],
        "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        "model__max_depth": [None, 5, 10],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        random_state=42,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    result_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "mean_fit_time",
        "params",
    ]
    return search.best_estimator_, results[result_columns].reset_index(drop=True)


def cross_validate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Measure model stability using training-only stratified cross-validation."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision"}
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        rows.append(
            {
                "model": name,
                "cv_folds": cv_folds,
                "roc_auc_mean": scores["test_roc_auc"].mean(),
                "roc_auc_std": scores["test_roc_auc"].std(),
                "average_precision_mean": scores["test_average_precision"].mean(),
                "average_precision_std": scores["test_average_precision"].std(),
                "fit_time_mean_seconds": scores["fit_time"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def compare_feature_engineering_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Quantify the cross-validated contribution of domain feature engineering."""
    estimator = HistGradientBoostingClassifier(random_state=42)
    models = {
        "Raw features": build_model_pipeline(
            clone(estimator),
            build_preprocessor(include_engineered=False),
            use_feature_engineering=False,
        ),
        "Raw + business features": build_model_pipeline(
            clone(estimator),
            build_preprocessor(include_engineered=True),
            use_feature_engineering=True,
        ),
    }
    return cross_validate_models(models, X_train, y_train, cv_folds=cv_folds)


def calibrate_model(model, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5):
    """Fit a sigmoid-calibrated copy of a selected model using training data only."""
    calibrated = CalibratedClassifierCV(
        estimator=clone(model),
        method="sigmoid",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def get_oof_probabilities(model, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5):
    """Generate out-of-fold probabilities for leakage-safe threshold selection."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    probabilities = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )
    return np.asarray(probabilities)[:, 1]


def select_best_model(metrics_df: pd.DataFrame) -> str:
    """Select the model with the highest ROC-AUC, then KS statistic."""
    required = {"model", "roc_auc", "ks"}
    if not required.issubset(metrics_df.columns):
        raise ValueError(f"metrics_df must contain columns: {sorted(required)}")
    ranked = metrics_df.sort_values(["roc_auc", "ks"], ascending=False)
    return str(ranked.iloc[0]["model"])
