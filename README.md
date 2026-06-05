# Credit Default Risk Modeling, Calibration and Decision Policy

An end-to-end credit-risk research and engineering project that predicts next-month default probability, validates model stability, converts scores into operational risk tiers, and selects a review threshold under an explicit business-cost assumption.

## Executive Summary

The project uses 30,000 anonymised credit-card client records from UCI. The final model is selected using training-only cross-validation, calibrated for probability interpretation, and evaluated once on an untouched 20% holdout set.

Validated full-run results with `random_state=42`:

| Result | Value |
|---|---:|
| Tuned HistGradientBoosting 5-fold CV ROC-AUC | **0.7869 ± 0.0042** |
| Final holdout ROC-AUC | **0.7821** |
| Final holdout KS | **0.4312** |
| Final holdout Average Precision / PR-AUC | **0.5585** |
| Top 10% default capture rate | **31.95%** |
| Top 10% lift | **3.20x** |
| Top 20% default capture rate | **51.24%** |
| Cost-optimised threshold recall | **74.91%** |
| Very High Risk observed default rate | **77.00%** |
| Low Risk observed default rate | **8.93%** |

The operational threshold is selected from out-of-fold training predictions, not from the test set. Under the documented assumption that a missed default costs five times as much as a false positive, the selected threshold is `0.16`, increasing holdout recall from `36.62%` to `74.91%`.

The calibrated model is selected as the operational probability model because it improves training out-of-fold Brier score from `0.13342` to `0.13314` and log loss from `0.42556` to `0.42460`. On the final holdout, the raw and calibrated models remain extremely close, which is reported transparently rather than treating calibration as automatically beneficial.

## Business Objective

- Estimate each client's probability of default in the next month.
- Rank customers by risk and concentrate monitoring resources.
- Convert predicted probabilities into Low, Medium, High, and Very High Risk tiers.
- Select an operating threshold that reflects asymmetric credit-risk costs.
- Explain the strongest model risk drivers for business review.

## Dataset

The project uses the public **UCI Default of Credit Card Clients** dataset:

- 30,000 client observations
- 23 raw input features
- Credit limit, demographics, six months of repayment status, bill amounts, and payment amounts
- Binary target: default payment next month

The preferred data-access method is automatic download through `ucimlrepo`. A manually downloaded CSV/XLS/XLSX file is also supported. Raw data is ignored by git.

## Research Design and Leakage Controls

1. Reserve a stratified 20% holdout set for final evaluation.
2. Perform feature engineering inside sklearn Pipelines.
3. Tune HistGradientBoosting using training-only stratified cross-validation.
4. Select the ranking model using training-set cross-validation results.
5. Calibrate the selected model using training data only.
6. Compare raw and calibrated probabilities using out-of-fold Brier score and log loss.
7. Generate out-of-fold probabilities for threshold selection.
8. Evaluate the final policy once on the untouched holdout set.

This design prevents test-set information from influencing preprocessing, tuning, model selection, calibration, or threshold selection.

## Business Feature Engineering

The pipeline expands 23 raw variables into 45 model inputs, including:

- Delinquent-month count and severe-delinquency count
- Maximum and recency-weighted delinquency status
- Repayment-status trend
- Recent, average, and maximum credit utilisation
- Bill growth relative to credit limit
- Payment-to-bill ratio and recent payment ratio
- Zero-payment months and bill-payment gap

Permutation importance identifies recency-weighted delinquency, recent repayment status, utilisation, bill volatility, credit limit, and payment coverage as major risk drivers.

## Models

- Logistic Regression
- Random Forest
- HistGradientBoosting
- Cross-validated tuned HistGradientBoosting
- Sigmoid-calibrated tuned HistGradientBoosting, selected by out-of-fold probability quality

## Evaluation Framework

Ranking and discrimination:

- ROC-AUC
- KS statistic
- Average Precision / PR-AUC
- Top-decile and top-20% default capture
- Lift and cumulative gains

Probability quality:

- Brier score
- Log loss
- Calibration curve

Decision policy:

- Precision, recall, F1, and confusion matrix
- Out-of-fold threshold analysis
- Explicit false-negative versus false-positive cost assumption

## Risk Segmentation

| Risk tier | Portfolio share | Observed default rate | Average predicted probability |
|---|---:|---:|---:|
| Low Risk | 50% | 8.93% | 9.03% |
| Medium Risk | 30% | 21.06% | 20.65% |
| High Risk | 15% | 49.89% | 51.29% |
| Very High Risk | 5% | 77.00% | 74.94% |

The monotonic increase in observed default rate demonstrates that the score supports meaningful portfolio segmentation.

## Project Structure

```text
credit-default-risk-modeling/
├── .github/workflows/ci.yml
├── README.md
├── requirements.txt
├── run_project.py
├── data/README.md
├── notebooks/01_credit_default_modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualization.py
├── tests/test_core.py
└── outputs/
    ├── figures/
    ├── models/
    └── analytical CSV reports
```

## Key Outputs

- `model_metrics.csv`: final holdout comparison with ranking, calibration, and capture metrics
- `cross_validation_metrics.csv`: mean and standard deviation across training folds
- `feature_engineering_impact.csv`: raw versus business-feature comparison
- `hyperparameter_search_results.csv`: reproducible tuning results
- `threshold_analysis.csv`: precision, recall, F1, and business cost across thresholds
- `threshold_metrics.csv`: default versus cost-optimised policy comparison
- `risk_tier_summary.csv`: portfolio segmentation report
- `decile_lift_summary.csv`: decile lift and cumulative default capture
- `permutation_feature_importance.csv`: model-driver analysis
- `test_predictions.csv`: holdout probabilities and tiers
- `calibration_comparison.csv`: training out-of-fold probability-quality comparison
- `models/best_operational_model.joblib`: reusable operational model artifact

The project also creates ROC, precision-recall, calibration, lift, threshold-cost, confusion-matrix, risk-tier, and explainability charts.

## How to Run

Windows / PyCharm:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_project.py
```

Quick smoke run:

```powershell
.\.venv\Scripts\python.exe run_project.py --fast
```

Notebook:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks/01_credit_default_modeling.ipynb
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Resume-Ready Chinese Bullets

- 基于 30,000 条信用卡客户数据构建端到端违约概率模型，在 sklearn Pipeline 内完成业务特征工程、交叉验证调参及概率校准，调优模型 5 折 ROC-AUC 达 `0.7869 ± 0.0042`，独立测试集 AUC 达 `0.7821`、KS 达 `0.4312`。
- 构建逾期频次、近期加权逾期程度、额度使用率、还款覆盖率和账单趋势等 22 个业务衍生特征，并通过 permutation importance 识别关键风险驱动因素。
- 基于训练集 out-of-fold 预测和非对称业务成本优化决策阈值，将独立测试集违约客户召回率由 `36.62%` 提升至 `74.91%`。
- 建立客户风险分层与 Lift/Capture 报告，最高风险 10% 客户覆盖 `31.95%` 的违约客户、Lift 达 `3.20x`，最高风险层实际违约率达 `77.00%`。

## Limitations and Governance

- The public dataset is cross-sectional; true out-of-time validation is not possible.
- The cost ratio used for threshold selection is illustrative and must be replaced by institution-specific economics.
- Real deployment requires fairness testing, stability monitoring, challenger models, governance approval, and compliance review.
- This project is for educational and portfolio purposes and must not be used directly for real credit decisions.
