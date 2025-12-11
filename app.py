# app.py
# Medicare Fraud Detection

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Medicare Fraud Detection",
    layout="wide",
)

sns.set(style="whitegrid")

RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Data paths (you can change the default to your own folder)
# -----------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path(__file__).parent / "data"

# Train files
TRAIN_PROVIDER = "Train-1542865627584.csv"
TRAIN_INPATIENT = "Train_Inpatientdata-1542865627584.csv"
TRAIN_OUTPATIENT = "Train_Outpatientdata-1542865627584.csv"
TRAIN_BENEFICIARY = "Train_Beneficiarydata-1542865627584.csv"


# -----------------------------------------------------------------------------
# Helper functions: IO + feature building + cleaning
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def build_provider_features(provider, inpatient, outpatient, beneficiary):

    def get_claim_amount_col(df, label):
        if "InscClaimAmtReimbursed" in df.columns:
            return "InscClaimAmtReimbursed"
        elif "ClaimAmount" in df.columns:
            return "ClaimAmount"
        else:
            raise KeyError(
                f"No claim amount column found in {label} "
                f"(expected 'InscClaimAmtReimbursed' or 'ClaimAmount')."
            )

    in_amt_col = get_claim_amount_col(inpatient, "inpatient")
    out_amt_col = get_claim_amount_col(outpatient, "outpatient")

    # Inpatient diagnosis non-null count per row
    inpatient = inpatient.copy()
    out_diag_cols_in = [
        c
        for c in inpatient.columns
        if "Diagnosis" in c or "ClmDiagnosisCode" in c or "DiagnosisGroupCode" in c
    ]
    if out_diag_cols_in:
        inpatient["in_diag_nonnull_row"] = inpatient[out_diag_cols_in].notna().sum(axis=1)
    else:
        inpatient["in_diag_nonnull_row"] = 0

    # Outpatient diagnosis non-null count per row
    outpatient = outpatient.copy()
    out_diag_cols_out = [
        c
        for c in outpatient.columns
        if "Diagnosis" in c or "ClmDiagnosisCode" in c or "DiagnosisGroupCode" in c
    ]
    if out_diag_cols_out:
        outpatient["out_diag_nonnull_row"] = outpatient[out_diag_cols_out].notna().sum(axis=1)
    else:
        outpatient["out_diag_nonnull_row"] = 0

    # Inpatient aggregation
    in_agg = inpatient.groupby("Provider").agg(
        in_claim_amt_sum=(in_amt_col, "sum"),
        in_claim_amt_mean=(in_amt_col, "mean"),
        in_claim_amt_max=(in_amt_col, "max"),
        in_unique_beneficiaries=("BeneID", "nunique"),
        in_diagnosis_nonnull_cnt=("in_diag_nonnull_row", "sum"),
        in_total_claims=("ClaimID", "count"),
    ).reset_index()

    # Outpatient aggregation
    out_agg = outpatient.groupby("Provider").agg(
        out_claim_amt_sum=(out_amt_col, "sum"),
        out_claim_amt_mean=(out_amt_col, "mean"),
        out_claim_amt_max=(out_amt_col, "max"),
        out_unique_beneficiaries=("BeneID", "nunique"),
        out_diagnosis_nonnull_cnt=("out_diag_nonnull_row", "sum"),
        out_total_claims=("ClaimID", "count"),
    ).reset_index()

    # Beneficiary chronic conditions → provider-level sum
    chronic_cols = [c for c in beneficiary.columns if c.startswith("ChronicCond_")]
    if chronic_cols:
        temp = beneficiary[["BeneID"] + chronic_cols].copy()
        temp["chronic_count"] = temp[chronic_cols].sum(axis=1)

        bene_provider = pd.concat(
            [
                inpatient[["Provider", "BeneID"]],
                outpatient[["Provider", "BeneID"]],
            ],
            ignore_index=True,
        ).drop_duplicates()
        bene_provider = bene_provider.merge(temp[["BeneID", "chronic_count"]], on="BeneID", how="left")
        bene_provider_agg = bene_provider.groupby("Provider").agg(
            chronic_cond_sum_total=("chronic_count", "sum")
        ).reset_index()
    else:
        bene_provider_agg = None

    df = provider.copy()
    df = df.merge(in_agg, on="Provider", how="left")
    df = df.merge(out_agg, on="Provider", how="left")

    if bene_provider_agg is not None:
        df = df.merge(bene_provider_agg, on="Provider", how="left")
    else:
        df["chronic_cond_sum_total"] = np.nan

    if "PotentialFraud" not in df.columns:
        raise KeyError("Column 'PotentialFraud' not found in provider data.")
    df["fraud_label"] = (df["PotentialFraud"] == "Yes").astype(int)

    df["in_avg_amt_per_claim"] = df["in_claim_amt_sum"] / df["in_total_claims"].replace(0, np.nan)
    df["out_avg_amt_per_claim"] = df["out_claim_amt_sum"] / df["out_total_claims"].replace(0, np.nan)

    df["total_claim_count"] = (
        df["in_total_claims"].fillna(0) + df["out_total_claims"].fillna(0)
    )
    df["total_unique_beneficiaries"] = (
        df["in_unique_beneficiaries"].fillna(0) + df["out_unique_beneficiaries"].fillna(0)
    )

    return df


def clean_provider_feature_table(df: pd.DataFrame, target_col: str = "fraud_label") -> pd.DataFrame:
    df = df.copy()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    keep_cols = ["Provider", "PotentialFraud", target_col]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = list(dict.fromkeys(keep_cols + num_cols))

    df = df[keep_cols]
    df = df.dropna(subset=[target_col])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


# -----------------------------------------------------------------------------
# Modeling helpers
# -----------------------------------------------------------------------------
def split_features_target(df, target_col="fraud_label"):
    non_features = {"fraud_label", "PotentialFraud", "Provider"}
    X = df[[c for c in df.columns if c not in non_features]].copy()
    y = df[target_col].copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    transformers = [
        (
            "num",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]
            ),
            num_cols,
        )
    ]
    return ColumnTransformer(transformers, remainder="drop")


def fit_with_smote(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pre = pipeline.named_steps["preproc"]
    clf = pipeline.named_steps["clf"]

    X_tr = pre.fit_transform(X_train)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_tr, y_train)

    clf.fit(X_res, y_res)
    return Pipeline(steps=[("preproc", pre), ("clf", clf)])


def evaluate_model(name: str, clf: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_prob)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "report": report,
        "fpr": fpr,
        "tpr": tpr,
    }


def generate_simple_report(df: pd.DataFrame, metrics: dict) -> str:
    total = len(df)
    fraud_rate = df["fraud_label"].mean() if "fraud_label" in df.columns else np.nan

    lines = f"""
Health Insurance Fraud Detection – Streamlit Report
==================================================

Total providers after cleaning: {total}
Estimated fraud rate: {fraud_rate:.3f}

Model: {metrics.get("model", "N/A")}
AUC: {metrics.get("auc", np.nan):.3f}
Precision: {metrics.get("precision", np.nan):.3f}
Recall: {metrics.get("recall", np.nan):.3f}
F1-score: {metrics.get("f1", np.nan):.3f}

Confusion Matrix:
{metrics.get("cm")}

Classification Report:
{metrics.get("report")}
"""
    return textwrap.dedent(lines)


# -----------------------------------------------------------------------------
# Session-state setup
# -----------------------------------------------------------------------------
if "provider" not in st.session_state:
    st.session_state.provider = None
    st.session_state.inpatient = None
    st.session_state.outpatient = None
    st.session_state.beneficiary = None
    st.session_state.provider_features = None
    st.session_state.feat = None
    st.session_state.last_metrics = None

# -----------------------------------------------------------------------------
# Layout – Tabs (as in the screenshot)
# -----------------------------------------------------------------------------
st.title("Health Insurance Fraud Detection")

(
    tab_overview,
    tab_load,
    tab_display,
    tab_clean,
    tab_models,
    tab_reports,
    tab_help,
) = st.tabs(
    [
        "Overview",
        "Load Data",
        "Display & Explore",
        "Clean & Preprocess",
        "Analytics & Models",
        "Reports & Full Pipeline",
        "Help",
    ]
)

# -----------------------------------------------------------------------------
# Overview tab
# -----------------------------------------------------------------------------
with tab_overview:
    st.markdown(
        """
**Health Insurance Fraud Detection**

This GUI lets you:

1. Load Medicare fraud CSVs and build a provider-level dataset.  
2. Display and explore data (tables, stats, histograms, boxplots).  
3. Clean and preprocess features for modeling.  
4. Train and evaluate models (LogReg / Random Forest / MLP).  
5. Generate a simple text report.  
6. Run the full pipeline end-to-end.

**OSEMN / Pipeline view**

`OBTAIN → SCRUB → EXPLORE → MODEL → INTERPRET`  
`Load → Clean → EDA → Train → Metrics & Report`
        """
    )

# -----------------------------------------------------------------------------
# Load Data tab
# -----------------------------------------------------------------------------
with tab_load:
    st.subheader("Load CSV Data and Build Provider-Level Features")

    data_dir_str = st.text_input(
        "Data directory containing the four train CSV files:",
        value=str(DEFAULT_DATA_DIR),
    )
    data_dir = Path(data_dir_str)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load CSV Data"):
            try:
                st.session_state.provider = read_csv_safely(data_dir / TRAIN_PROVIDER)
                st.session_state.inpatient = read_csv_safely(data_dir / TRAIN_INPATIENT)
                st.session_state.outpatient = read_csv_safely(data_dir / TRAIN_OUTPATIENT)
                st.session_state.beneficiary = read_csv_safely(data_dir / TRAIN_BENEFICIARY)

                st.success("Raw CSVs loaded successfully.")
                st.write("Provider:", st.session_state.provider.shape)
                st.write("Inpatient:", st.session_state.inpatient.shape)
                st.write("Outpatient:", st.session_state.outpatient.shape)
                st.write("Beneficiary:", st.session_state.beneficiary.shape)

                st.dataframe(st.session_state.provider.head())
            except Exception as e:
                st.error(f"Error loading CSVs: {e}")

    with col2:
        if st.button("Build Provider-Level Features"):
            try:
                if st.session_state.provider is None:
                    st.warning("Load CSV data first.")
                else:
                    pf = build_provider_features(
                        st.session_state.provider,
                        st.session_state.inpatient,
                        st.session_state.outpatient,
                        st.session_state.beneficiary,
                    )
                    st.session_state.provider_features = pf
                    st.success(f"Provider-level dataset built. Shape: {pf.shape}")
                    st.dataframe(pf.head())
            except Exception as e:
                st.error(f"Error building provider features: {e}")

# -----------------------------------------------------------------------------
# Display & Explore tab
# -----------------------------------------------------------------------------
with tab_display:
    st.subheader("Display & Explore Data")

    df_for_view = None
    label = ""

    if st.session_state.feat is not None:
        df_for_view = st.session_state.feat
        label = "Cleaned feature table"
    elif st.session_state.provider_features is not None:
        df_for_view = st.session_state.provider_features
        label = "Provider-level dataset"

    if df_for_view is None:
        st.info("No data available. Load and build provider-level features first.")
    else:
        if st.button("Show Preview & Summary"):
            st.write(f"Preview of {label}:")
            st.dataframe(df_for_view.head())
            st.write("Numeric summary:")
            st.dataframe(df_for_view.select_dtypes(include=[np.number]).describe())

        numeric_cols = df_for_view.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col_sel = st.selectbox("Numeric column for plots:", numeric_cols)

            c1, c2 = st.columns(2)

            with c1:
                st.write("Histogram")
                fig, ax = plt.subplots(figsize=(6, 4))
                if "fraud_label" in df_for_view.columns:
                    data0 = df_for_view[df_for_view["fraud_label"] == 0][col_sel].dropna()
                    data1 = df_for_view[df_for_view["fraud_label"] == 1][col_sel].dropna()
                    ax.hist(data0, bins=30, alpha=0.6, label="Not Fraud")
                    ax.hist(data1, bins=30, alpha=0.6, label="Fraud")
                    ax.legend()
                    ax.set_title(f"Histogram of {col_sel} by fraud_label")
                else:
                    ax.hist(df_for_view[col_sel].dropna(), bins=30)
                    ax.set_title(f"Histogram of {col_sel}")
                ax.set_xlabel(col_sel)
                st.pyplot(fig)

            with c2:
                st.write("Boxplot by fraud_label")
                if "fraud_label" in df_for_view.columns:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    data0 = df_for_view[df_for_view["fraud_label"] == 0][col_sel].dropna()
                    data1 = df_for_view[df_for_view["fraud_label"] == 1][col_sel].dropna()
                    ax.boxplot([data0, data1], labels=["Not Fraud", "Fraud"])
                    ax.set_title(f"Boxplot of {col_sel} by fraud_label")
                    st.pyplot(fig)
                else:
                    st.info("fraud_label not found; boxplot by label not available.")
        else:
            st.info("No numeric columns found for plotting.")

# -----------------------------------------------------------------------------
# Clean & Preprocess tab
# -----------------------------------------------------------------------------
with tab_clean:
    st.subheader("Clean Data & Build Feature Table")

    if st.button("Clean Data & Build Feature Table"):
        if st.session_state.provider_features is None:
            st.warning("Build provider-level dataset first on the Load Data tab.")
        else:
            try:
                clean_df = clean_provider_feature_table(
                    st.session_state.provider_features, target_col="fraud_label"
                )
                st.session_state.feat = clean_df
                st.success(f"Cleaned feature table shape: {clean_df.shape}")
                st.dataframe(clean_df.head())

                st.write("Fraud label distribution (counts):")
                st.write(clean_df["fraud_label"].value_counts())
                st.write("Fraud label distribution (normalized):")
                st.write(clean_df["fraud_label"].value_counts(normalize=True))
            except Exception as e:
                st.error(f"Error cleaning data: {e}")

# -----------------------------------------------------------------------------
# Analytics & Models tab
# -----------------------------------------------------------------------------
with tab_models:
    st.subheader("Train Models with SMOTE and Evaluate")

    if st.session_state.feat is None:
        st.info("Clean feature table not available. Run the Clean & Preprocess step first.")
    else:
        model_name = st.selectbox(
            "Select model:",
            ["Logistic Regression", "Random Forest", "MLP Neural Network"],
        )

        if st.button("Train & Evaluate"):
            df = st.session_state.feat
            X, y = split_features_target(df, target_col="fraud_label")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
            )
            preproc = build_preprocessor(X_train)

            if model_name == "Logistic Regression":
                clf = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
            elif model_name == "Random Forest":
                clf = RandomForestClassifier(
                    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
                )
            else:
                clf = MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    max_iter=800,
                    early_stopping=True,
                    n_iter_no_change=10,
                    alpha=1e-3,
                    random_state=RANDOM_STATE,
                )

            pipe = Pipeline(steps=[("preproc", preproc), ("clf", clf)])
            fitted = fit_with_smote(pipe, X_train, y_train)
            metrics = evaluate_model(model_name, fitted, X_test, y_test)
            st.session_state.last_metrics = metrics

            st.markdown("**Metrics**")
            st.write(f"AUC: {metrics['auc']:.3f}")
            st.write(f"Precision: {metrics['precision']:.3f}")
            st.write(f"Recall: {metrics['recall']:.3f}")
            st.write(f"F1-score: {metrics['f1']:.3f}")

            st.markdown("**Confusion Matrix**")
            st.write(pd.DataFrame(
                metrics["cm"],
                index=["Actual 0", "Actual 1"],
                columns=["Pred 0", "Pred 1"],
            ))

            st.markdown("**Classification Report**")
            st.text(metrics["report"])

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(metrics["fpr"], metrics["tpr"], label=f"AUC={metrics['auc']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve – {model_name}")
            ax.legend()
            st.pyplot(fig)

# -----------------------------------------------------------------------------
# Reports & Full Pipeline tab
# -----------------------------------------------------------------------------
with tab_reports:
    st.subheader("Full Pipeline and Text Report")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Full Pipeline (Random Forest)"):
            try:
                # Reload and run everything in one go
                data_dir = Path(data_dir_str)

                provider = read_csv_safely(data_dir / TRAIN_PROVIDER)
                inpatient = read_csv_safely(data_dir / TRAIN_INPATIENT)
                outpatient = read_csv_safely(data_dir / TRAIN_OUTPATIENT)
                beneficiary = read_csv_safely(data_dir / TRAIN_BENEFICIARY)

                pf = build_provider_features(provider, inpatient, outpatient, beneficiary)
                feat = clean_provider_feature_table(pf, target_col="fraud_label")

                X, y = split_features_target(feat, target_col="fraud_label")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
                )
                preproc = build_preprocessor(X_train)
                clf = RandomForestClassifier(
                    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
                )
                pipe = Pipeline(steps=[("preproc", preproc), ("clf", clf)])
                fitted = fit_with_smote(pipe, X_train, y_train)
                metrics = evaluate_model("Random Forest", fitted, X_test, y_test)

                st.session_state.provider = provider
                st.session_state.inpatient = inpatient
                st.session_state.outpatient = outpatient
                st.session_state.beneficiary = beneficiary
                st.session_state.provider_features = pf
                st.session_state.feat = feat
                st.session_state.last_metrics = metrics

                st.success("Full pipeline completed successfully.")
                st.write("Random Forest metrics:")
                st.write(f"AUC: {metrics['auc']:.3f}")
                st.write(f"Precision: {metrics['precision']:.3f}")
                st.write(f"Recall: {metrics['recall']:.3f}")
                st.write(f"F1-score: {metrics['f1']:.3f}")
            except Exception as e:
                st.error(f"Error running full pipeline: {e}")

    with col2:
        if st.button("Generate Text Report"):
            df = st.session_state.feat
            metrics = st.session_state.last_metrics
            if df is None or metrics is None:
                st.warning("Need cleaned data and at least one trained model.")
            else:
                report_text = generate_simple_report(df, metrics)
                st.text_area("Report", report_text, height=400)
                st.download_button(
                    "Download report as .txt",
                    data=report_text,
                    file_name="medicare_fraud_report.txt",
                    mime="text/plain",
                )

# -----------------------------------------------------------------------------
# Help tab
# -----------------------------------------------------------------------------
with tab_help:
    st.markdown(
        """
# Help – How to Use the Medicare Fraud Detection Tool

Use the tabs from left to right:

1. **Overview** – High-level description of the system and pipeline.  
2. **Load Data** – Load raw CSVs and build provider-level features.  
3. **Display & Explore** – Preview tables, summary statistics, histograms, and boxplots.  
4. **Clean & Preprocess** – Build the cleaned feature table ready for modeling.  
5. **Analytics & Models** – Train Logistic Regression, Random Forest, or MLP with SMOTE and view metrics/ROC curves.  
6. **Reports & Full Pipeline** – Run the full pipeline end-to-end and generate a text report.  
7. **Help** – This page.

Recommended order:  
**Overview → Load Data → Display & Explore → Clean & Preprocess → Analytics & Models → Reports & Full Pipeline → Help (anytime).**
        """
    )
