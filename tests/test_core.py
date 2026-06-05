"""Small offline tests for the core credit-risk workflow."""

import numpy as np
import pandas as pd

from src.data_preprocessing import clean_credit_data, split_features_target
from src.evaluation import create_risk_tiers, select_cost_optimal_threshold, threshold_analysis
from src.feature_engineering import ENGINEERED_FEATURES, add_business_features


def make_sample_data(n_rows=100):
    rng = np.random.default_rng(42)
    data = {
        "credit_limit": rng.integers(10_000, 500_000, n_rows),
        "sex": rng.integers(1, 3, n_rows),
        "education": rng.integers(0, 7, n_rows),
        "marriage": rng.integers(0, 4, n_rows),
        "age": rng.integers(21, 70, n_rows),
    }
    for month in ["sep", "aug", "jul", "jun", "may", "apr"]:
        data[f"repay_status_{month}"] = rng.integers(-2, 5, n_rows)
        data[f"bill_amt_{month}"] = rng.integers(-2_000, 300_000, n_rows)
        data[f"pay_amt_{month}"] = rng.integers(0, 50_000, n_rows)
    data["default_next_month"] = rng.integers(0, 2, n_rows)
    return pd.DataFrame(data)


def test_cleaning_and_feature_engineering_are_complete():
    cleaned = clean_credit_data(make_sample_data())
    X, y = split_features_target(cleaned)
    featured = add_business_features(X)

    assert set(ENGINEERED_FEATURES).issubset(featured.columns)
    assert featured.isna().sum().sum() == 0
    assert set(y.unique()).issubset({0, 1})
    assert set(cleaned["education"].unique()).isdisjoint({0, 5, 6})
    assert 0 not in set(cleaned["marriage"].unique())


def test_risk_tiers_follow_requested_portfolio_shares():
    tiers = create_risk_tiers(np.linspace(0.01, 0.99, 100))
    counts = tiers.value_counts()

    assert counts["Low Risk"] == 50
    assert counts["Medium Risk"] == 30
    assert counts["High Risk"] == 15
    assert counts["Very High Risk"] == 5


def test_cost_optimal_threshold_is_returned_from_analysis():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.20, 0.35, 0.30, 0.55, 0.75, 0.90])
    analysis = threshold_analysis(y_true, y_prob, false_negative_cost=5, false_positive_cost=1)
    threshold = select_cost_optimal_threshold(analysis)

    assert threshold in set(analysis["threshold"])
    assert 0.05 <= threshold <= 0.80
