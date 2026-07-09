# TAVI Residual MR & Mortality Prediction Pipeline


A complete, reproducible machine learning pipeline for predicting residual mitral regurgitation (MR) and landmark mortality after Transcatheter Aortic Valve Implantation (TAVI).

## Overview

| Task | Outcome | Predictors | Method |
|------|---------|------------|--------|
| Residual MR | Moderate+ MR at follow-up echo | Baseline clinical + pre-procedural echo | XGBoost (nested CV) |
| Baseline 3-yr Mortality | Death within 3 years of TAVI | Baseline clinical + pre-procedural echo | XGBoost (nested CV) |
| Landmark 2-yr Mortality | Death within 2 years post-landmark | Baseline + follow-up echo (delta features) | XGBoost (nested CV) |
| Survival Analysis | Time-to-death | Baseline / Landmark covariates | Penalized Cox PH |

## Key Features

- Leakage-proof nested cross-validation (5 outer x 3 inner folds)
- Optuna Bayesian optimization (50 trials per inner fold) for XGBoost
- SHAP explainability with mean|SHAP| feature importance
- Penalized Cox PH survival models with forest plots
- Full calibration assessment (reliability diagrams, ECE, Brier score)
- Landmark analysis at 1 year with delta-echo features
- All preprocessing inside CV pipelines - zero data leakage
- Reproducible: fixed seeds, exact CV splits, Optuna TPE sampler


# Clone and install
git clone https://github.com/PooryaAmini/TAVI_MR.git
cd TAVI_MR
pip install -r requirements.txt


# Run full pipeline
python tavimr_analysis_v2.py
`

## Data Requirements

Place TAVI_MR_paper_R1.xlsx (sheet: TAVI_MR_paper) in the project root. The script auto-detects common column name variants (e.g., EF/LVEF, MR_Grade/MR_grade, followup_days/Followup_Days).

Expected columns (flexible naming accepted):
- Demographics: Age, Gender, BSA
- Procedure: Pros Size, Approach, Valve_Type, Valve_Size
- Echo: pre_MR, pre_AR, pre_LVEF, pre_NYHA, FU_MR, FU_AR, FU_LVEF, FU_NYHA
- Comorbidities: CAD, HBP, DM, COPD, AF, PVD, CKD, etc.
- Outcomes: Status_Death, Death_Date, Last FUP Date, TAVI Date, Final_Population

See METHODS.md for full methodology details.

## Outputs

- analysis_summary_v2.xlsx       # Summary workbook
- oof_preds_*.csv                # Out-of-fold predictions
- feature_importance_*.csv       # Mean |SHAP| importance
- shap_summary_*.png             # SHAP beeswarm/bar plots
- calibration_*.png              # Calibration curves
- roc_pr_*.png                   # ROC & PR curves
- cox_summary_*.csv              # Cox PH coefficients
- cox_forest_*.png               # Forest plots

## Methods Summary

- Study Design: Retrospective TAVI registry (Mendeley Data: h773rp5czz.1)
- Population: Final analytic cohort after predefined filters
- Primary Outcome: Residual MR >= moderate at follow-up echo
- Secondary: Landmark mortality at 1-year (2-year horizon)
- Models: XGBoost with nested CV (Optuna, 50 trials/inner fold)
- Validation: Stratified 5-fold outer CV, 3-fold inner CV
- Calibration: Isotonic regression on OOF predictions
- Explainability: SHAP TreeExplainer, mean|SHAP| across outer folds
- Survival: Penalized Cox (l2=0.1), standardized continuous vars, one-hot categorical
- Software: Python 3.9+, scikit-learn, XGBoost, Optuna, SHAP, lifelines

See METHODS.md for full methodological details.

## Reproducibility

- All random states fixed to 42 (Python, NumPy, XGBoost, Optuna, SHAP, CV splits)
- Optuna: TPESampler(seed=42), MedianPruner(n_warmup_steps=10)
- Nested CV splits deterministic via StratifiedKFold(shuffle=True, random_state=42)
- SHAP: tree_method=exact for reproducibility

## Requirements

python>=3.9
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
optuna>=3.4
shap>=0.42
lifelines>=0.27
matplotlib>=3.7
seaborn>=0.12
openpyxl>=3.1
optuna-integration>=3.4
scipy>=1.11
joblib>=1.3

Install: pip install -r requirements.txt


Data source: Muratori et al., Data for: Mitral valve regurgitation in patients undergoing TAVI, Mendeley Data, V1, doi: 10.17632/h773rp5czz.1

## License

MIT License - see LICENSE for details.

