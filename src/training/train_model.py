"""
train_model.py

Fokus:
- Load data dari Spark
- Train 2 model (LogReg & Decision Tree)
- Pilih best model berdasarkan PR-AUC test
- Log & register best model ke MLflow (minimal setup, sama seperti notebook yang sudah jalan)

Tidak ada:
- batch scoring
- set_experiment / set_tracking_uri
"""

import argparse
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature


# ==============================
# Logging
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_model")


# ==============================
# Defaults (sesuai notebook-mu)
# ==============================
DEFAULT_FEATURE_TABLE = "cross_sell_insurance.01_feature_staging.stage2_clean_feature_table"
DEFAULT_TARGET_COL = "is_target_customer"
DEFAULT_REGISTERED_MODEL_NAME = "ml_model.bebas_aksi_classifier_proba"  # UC: catalog.schema.name


def get_env_or_default(env_name: str, default: str) -> str:
    """Baca env var, kalau 'None'/'none' dianggap kosong dan pakai default."""
    val = os.getenv(env_name)
    if val is None:
        return default
    if val.strip().lower() == "none":
        return default
    return val


# ==============================
# Argparse (optional, dengan default)
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Cross-Sell Insurance Propensity Model"
    )

    parser.add_argument(
        "--feature-table",
        required=False,
        default=get_env_or_default("FEATURE_TABLE", DEFAULT_FEATURE_TABLE),
        help="Spark table name for training data.",
    )

    parser.add_argument(
        "--target-col",
        required=False,
        default=get_env_or_default("TARGET_COL", DEFAULT_TARGET_COL),
        help="Target column name.",
    )

    parser.add_argument(
        "--registered-model-name",
        required=False,
        default=get_env_or_default(
            "REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL_NAME
        ),
        help="Registered model name in MLflow Model Registry (Unity Catalog).",
    )

    # Penting: supaya tidak error di notebook karena argumen -f
    args, _ = parser.parse_known_args()
    return args


# ==============================
# MLflow wrapper
# ==============================
class ProbaWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper supaya predict() mengembalikan probability (kolom 1).
    """

    def __init__(self, sk_model: Pipeline):
        self.sk_model = sk_model

    def predict(self, context: Any, model_input: pd.DataFrame) -> np.ndarray:
        proba = self.sk_model.predict_proba(model_input)[:, 1]
        return proba


# ==============================
# Load data
# ==============================
def load_data(table_name: str) -> pd.DataFrame:
    spark = SparkSession.builder.getOrCreate()
    logger.info(f"Loading data from Spark table: {table_name}")
    df_spark = spark.table(table_name)
    df = df_spark.toPandas()
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


# ==============================
# Build pipelines
# ==============================
def build_pipelines(X_train: pd.DataFrame):
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    logger.info(f"Numeric columns    : {numeric_cols}")
    logger.info(f"Categorical columns: {categorical_cols}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="0")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    logreg_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "clf",
                LogisticRegression(
                    max_iter=800,
                    class_weight="balanced",
                    n_jobs=-1,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    tree_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=50,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return logreg_pipeline, tree_pipeline


# ==============================
# Train + pilih best
# ==============================
def train_and_select_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    logreg_pipeline, tree_pipeline = build_pipelines(X_train)

    logger.info("Training Logistic Regression...")
    logreg_pipeline.fit(X_train, y_train)

    logger.info("Training Decision Tree...")
    tree_pipeline.fit(X_train, y_train)

    y_proba_log = logreg_pipeline.predict_proba(X_test)[:, 1]
    y_proba_tree = tree_pipeline.predict_proba(X_test)[:, 1]

    metrics: Dict[str, Dict[str, float]] = {
        "logistic": {
            "roc_auc": roc_auc_score(y_test, y_proba_log),
            "pr_auc": average_precision_score(y_test, y_proba_log),
        },
        "tree": {
            "roc_auc": roc_auc_score(y_test, y_proba_tree),
            "pr_auc": average_precision_score(y_test, y_proba_tree),
        },
    }

    logger.info("Test metrics:")
    for name, m in metrics.items():
        logger.info(
            f"  {name}: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}"
        )

    best_name = max(metrics, key=lambda m: metrics[m]["pr_auc"])
    best_model = logreg_pipeline if best_name == "logistic" else tree_pipeline

    logger.info(f"Best model selected (by PR-AUC): {best_name}")

    return best_model, best_name, metrics[best_name]


# ==============================
# Log & register ke MLflow (minimal)
# ==============================
def log_and_register_model(
    best_model: Pipeline,
    best_model_name: str,
    metrics: Dict[str, float],
    registered_model_name: str,
    X_train: pd.DataFrame,
):
    """
    Log & register model ke MLflow, minimal:
    - set_registry_uri("databricks-uc")
    - start_run()
    - pyfunc.log_model(..., registered_model_name=...)
    """
    # Sama seperti snippet yang sudah terbukti jalan:
    mlflow.set_registry_uri("databricks-uc")

    wrapped_model = ProbaWrapper(best_model)
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    signature = infer_signature(X_train, y_train_proba)
    input_example = X_train.head(5)

    logger.info(f"Registering model to MLflow as: {registered_model_name}")

    with mlflow.start_run():
        # Opsional: log sedikit info, tapi jangan yang ribet-ribet
        mlflow.set_tag("model_family", best_model_name)
        mlflow.log_metric("test_roc_auc", metrics["roc_auc"])
        mlflow.log_metric("test_pr_auc", metrics["pr_auc"])

        mlflow.pyfunc.log_model(
            python_model=wrapped_model,
            artifact_path="bebas_aksi_proba_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

    logger.info("Model logged & registered successfully.")


# ==============================
# Main
# ==============================
def main():
    args = parse_args()

    logger.info("===== Training config =====")
    logger.info(f"Feature table         : {args.feature_table}")
    logger.info(f"Target column         : {args.target_col}")
    logger.info(f"Registered model name : {args.registered_model_name}")

    df = load_data(args.feature_table)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in data!")

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    counts = y.value_counts()
    pct = y.value_counts(normalize=True) * 100
    logger.info("Class distribution:")
    logger.info("\n%s", pd.DataFrame({"count": counts, "pct": pct.round(4)}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model, best_name, best_metrics = train_and_select_model(
        X_train, X_test, y_train, y_test
    )

    log_and_register_model(
        best_model=best_model,
        best_model_name=best_name,
        metrics=best_metrics,
        registered_model_name=args.registered_model_name,
        X_train=X_train,
    )

    logger.info("Training job completed successfully.")


if __name__ == "__main__":
    main()