"""Data loading, cleaning, and preprocessing utilities."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import ENGINEERED_FEATURES


LOCAL_FILENAMES = (
    "default_of_credit_card_clients.csv",
    "default_of_credit_card_clients.xls",
    "default_of_credit_card_clients.xlsx",
)

COLUMN_MAPPING = {
    "X1": "credit_limit",
    "X2": "sex",
    "X3": "education",
    "X4": "marriage",
    "X5": "age",
    "X6": "repay_status_sep",
    "X7": "repay_status_aug",
    "X8": "repay_status_jul",
    "X9": "repay_status_jun",
    "X10": "repay_status_may",
    "X11": "repay_status_apr",
    "X12": "bill_amt_sep",
    "X13": "bill_amt_aug",
    "X14": "bill_amt_jul",
    "X15": "bill_amt_jun",
    "X16": "bill_amt_may",
    "X17": "bill_amt_apr",
    "X18": "pay_amt_sep",
    "X19": "pay_amt_aug",
    "X20": "pay_amt_jul",
    "X21": "pay_amt_jun",
    "X22": "pay_amt_may",
    "X23": "pay_amt_apr",
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "repay_status_sep",
    "PAY_2": "repay_status_aug",
    "PAY_3": "repay_status_jul",
    "PAY_4": "repay_status_jun",
    "PAY_5": "repay_status_may",
    "PAY_6": "repay_status_apr",
    "BILL_AMT1": "bill_amt_sep",
    "BILL_AMT2": "bill_amt_aug",
    "BILL_AMT3": "bill_amt_jul",
    "BILL_AMT4": "bill_amt_jun",
    "BILL_AMT5": "bill_amt_may",
    "BILL_AMT6": "bill_amt_apr",
    "PAY_AMT1": "pay_amt_sep",
    "PAY_AMT2": "pay_amt_aug",
    "PAY_AMT3": "pay_amt_jul",
    "PAY_AMT4": "pay_amt_jun",
    "PAY_AMT5": "pay_amt_may",
    "PAY_AMT6": "pay_amt_apr",
    "DEFAULT PAYMENT NEXT MONTH": "default_next_month",
    "Y": "default_next_month",
    "TARGET": "default_next_month",
}


def find_local_data_file(data_dir: Path) -> Optional[Path]:
    """Return the first supported local dataset path, if one exists."""
    data_dir = Path(data_dir)
    for filename in LOCAL_FILENAMES:
        path = data_dir / filename
        if path.exists():
            return path
    return None


def _read_data_file(path: Path) -> pd.DataFrame:
    """Read CSV or Excel data and handle the UCI workbook's extra title row."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported data format: {path.suffix}")

    standardized = standardize_column_names(df)
    if "default_next_month" not in standardized.columns and path.suffix.lower() in {".xls", ".xlsx"}:
        standardized = standardize_column_names(pd.read_excel(path, header=1))
    return standardized


def fetch_uci_credit_default_data(data_dir: Path) -> pd.DataFrame:
    """Download the UCI dataset, cache it as CSV, and return it."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "default_of_credit_card_clients.csv"

    if csv_path.exists():
        return standardize_column_names(pd.read_csv(csv_path))

    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=350)
        features = dataset.data.features.reset_index(drop=True)
        targets = dataset.data.targets.reset_index(drop=True)
        target = targets.iloc[:, 0].rename("default_next_month")
        df = pd.concat([features, target], axis=1)
        df = standardize_column_names(df)
        df.to_csv(csv_path, index=False)
        return df
    except Exception as exc:
        raise RuntimeError(
            "Please download the UCI Default of Credit Card Clients dataset "
            "and place it under data/."
        ) from exc


def load_credit_default_data(data_dir: Path) -> pd.DataFrame:
    """Load a local dataset when available, otherwise download it from UCI."""
    data_dir = Path(data_dir)
    local_path = find_local_data_file(data_dir)
    if local_path is not None:
        df = _read_data_file(local_path)
    else:
        df = fetch_uci_credit_default_data(data_dir)

    if "default_next_month" not in df.columns:
        raise ValueError(
            "Target column not found. Expected 'default payment next month', 'Y', or 'target'."
        )
    return standardize_column_names(df)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw UCI fields to clear, analysis-friendly names."""
    renamed = df.copy()
    normalized_mapping = {key.upper(): value for key, value in COLUMN_MAPPING.items()}
    column_names = {}

    for column in renamed.columns:
        clean_name = str(column).strip()
        lookup_name = clean_name.upper()
        column_names[column] = normalized_mapping.get(lookup_name, clean_name)

    return renamed.rename(columns=column_names)


def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean categories, missing values, identifiers, and the binary target."""
    cleaned = standardize_column_names(df)
    id_columns = [column for column in cleaned.columns if str(column).strip().lower() == "id"]
    cleaned = cleaned.drop(columns=id_columns, errors="ignore")

    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "education" in cleaned.columns:
        cleaned["education"] = cleaned["education"].replace({0: 4, 5: 4, 6: 4})
    if "marriage" in cleaned.columns:
        cleaned["marriage"] = cleaned["marriage"].replace({0: 3})

    categorical_features = [column for column in ["sex", "education", "marriage"] if column in cleaned]
    for column in cleaned.columns:
        if not cleaned[column].isna().any():
            continue
        if column in categorical_features:
            mode = cleaned[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else 0
        else:
            fill_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(fill_value)

    if "default_next_month" not in cleaned.columns:
        raise ValueError("The cleaned dataset must contain default_next_month.")

    target_values = set(cleaned["default_next_month"].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError("default_next_month must contain only binary values 0 and 1.")
    cleaned["default_next_month"] = cleaned["default_next_month"].astype(int)

    return cleaned


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into model features and target."""
    X = df.drop(columns="default_next_month")
    y = df["default_next_month"].astype(int)
    return X, y


def get_feature_groups(include_engineered: bool = True) -> Tuple[list[str], list[str]]:
    """Return the numeric and categorical feature groups used by the model."""
    categorical_features = ["sex", "education", "marriage"]
    numeric_features = [
        "credit_limit",
        "age",
        "repay_status_sep",
        "repay_status_aug",
        "repay_status_jul",
        "repay_status_jun",
        "repay_status_may",
        "repay_status_apr",
        "bill_amt_sep",
        "bill_amt_aug",
        "bill_amt_jul",
        "bill_amt_jun",
        "bill_amt_may",
        "bill_amt_apr",
        "pay_amt_sep",
        "pay_amt_aug",
        "pay_amt_jul",
        "pay_amt_jun",
        "pay_amt_may",
        "pay_amt_apr",
    ]
    if include_engineered:
        numeric_features.extend(ENGINEERED_FEATURES)
    return numeric_features, categorical_features


def build_preprocessor(include_engineered: bool = True) -> ColumnTransformer:
    """Build leakage-safe preprocessing for use inside sklearn Pipelines."""
    numeric_features, categorical_features = get_feature_groups(include_engineered=include_engineered)
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        sparse_threshold=0.0,
    )
