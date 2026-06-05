"""Domain-informed feature engineering for credit default risk."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


REPAY_STATUS_FEATURES = [
    "repay_status_sep",
    "repay_status_aug",
    "repay_status_jul",
    "repay_status_jun",
    "repay_status_may",
    "repay_status_apr",
]

BILL_AMOUNT_FEATURES = [
    "bill_amt_sep",
    "bill_amt_aug",
    "bill_amt_jul",
    "bill_amt_jun",
    "bill_amt_may",
    "bill_amt_apr",
]

PAYMENT_AMOUNT_FEATURES = [
    "pay_amt_sep",
    "pay_amt_aug",
    "pay_amt_jul",
    "pay_amt_jun",
    "pay_amt_may",
    "pay_amt_apr",
]

ENGINEERED_FEATURES = [
    "delinquent_months",
    "severe_delinquent_months",
    "max_delinquency_status",
    "mean_positive_delinquency",
    "recent_delinquency_weighted",
    "repayment_status_trend",
    "avg_bill_amt",
    "max_bill_amt",
    "bill_amt_std",
    "total_positive_bill_amt",
    "bill_growth_to_limit",
    "recent_credit_utilization",
    "avg_credit_utilization",
    "max_credit_utilization",
    "avg_pay_amt",
    "max_pay_amt",
    "pay_amt_std",
    "total_pay_amt",
    "payment_to_bill_ratio",
    "recent_payment_ratio",
    "zero_payment_months",
    "bill_payment_gap_to_limit",
]


def _safe_divide(numerator, denominator, lower=None, upper=None):
    denominator = np.asarray(denominator, dtype=float)
    safe_denominator = np.where(np.abs(denominator) < 1.0, 1.0, denominator)
    result = np.asarray(numerator, dtype=float) / safe_denominator
    if lower is not None or upper is not None:
        result = np.clip(result, lower, upper)
    return result


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable repayment, utilisation, and payment-capacity features."""
    featured = df.copy()

    repay = featured[REPAY_STATUS_FEATURES]
    positive_repay = repay.clip(lower=0)
    bills = featured[BILL_AMOUNT_FEATURES]
    positive_bills = bills.clip(lower=0)
    payments = featured[PAYMENT_AMOUNT_FEATURES].clip(lower=0)
    credit_limit = featured["credit_limit"].clip(lower=1)

    featured["delinquent_months"] = (repay > 0).sum(axis=1)
    featured["severe_delinquent_months"] = (repay > 1).sum(axis=1)
    featured["max_delinquency_status"] = positive_repay.max(axis=1)
    featured["mean_positive_delinquency"] = positive_repay.mean(axis=1)
    featured["recent_delinquency_weighted"] = positive_repay.mul([6, 5, 4, 3, 2, 1]).sum(axis=1) / 21
    featured["repayment_status_trend"] = repay["repay_status_sep"] - repay["repay_status_apr"]

    featured["avg_bill_amt"] = bills.mean(axis=1)
    featured["max_bill_amt"] = bills.max(axis=1)
    featured["bill_amt_std"] = bills.std(axis=1)
    featured["total_positive_bill_amt"] = positive_bills.sum(axis=1)
    featured["bill_growth_to_limit"] = _safe_divide(
        bills["bill_amt_sep"] - bills["bill_amt_apr"], credit_limit, -5, 5
    )

    utilisation = bills.div(credit_limit, axis=0).clip(lower=-1, upper=5)
    featured["recent_credit_utilization"] = utilisation["bill_amt_sep"]
    featured["avg_credit_utilization"] = utilisation.mean(axis=1)
    featured["max_credit_utilization"] = utilisation.max(axis=1)

    featured["avg_pay_amt"] = payments.mean(axis=1)
    featured["max_pay_amt"] = payments.max(axis=1)
    featured["pay_amt_std"] = payments.std(axis=1)
    featured["total_pay_amt"] = payments.sum(axis=1)
    featured["payment_to_bill_ratio"] = _safe_divide(
        featured["total_pay_amt"], featured["total_positive_bill_amt"], 0, 5
    )
    featured["recent_payment_ratio"] = _safe_divide(
        payments["pay_amt_sep"], positive_bills["bill_amt_aug"], 0, 5
    )
    featured["zero_payment_months"] = (payments == 0).sum(axis=1)
    featured["bill_payment_gap_to_limit"] = _safe_divide(
        featured["total_positive_bill_amt"] - featured["total_pay_amt"],
        credit_limit,
        -5,
        20,
    )

    return featured


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for leakage-safe business features."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = np.asarray(add_business_features(X.head(1)).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_business_features(X)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
