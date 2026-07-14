from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    RandomForestClassifier = None
    SimpleImputer = None
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None
    train_test_split = None
    accuracy_score = None
    average_precision_score = None
    confusion_matrix = None
    f1_score = None
    precision_recall_curve = None
    precision_score = None
    recall_score = None
    roc_auc_score = None

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None


FEATURE_COLUMNS = ["Time", "Amount", *[f"V{index}" for index in range(1, 29)]]
ENGINEERED_COLUMNS = ["LogAmount", "Hour", "DayIndex"]
MODEL_FEATURES = FEATURE_COLUMNS + ENGINEERED_COLUMNS


@dataclass
class DatasetBundle:
    raw: pd.DataFrame
    clean: pd.DataFrame
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Credit card fraud detection with baseline ML and deep learning.")
    parser.add_argument(
        "--input",
        default=str(project_root / "data" / "creditcard.csv.zip"),
        help="Path to creditcard.csv or creditcard.csv.zip.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "output"),
        help="Directory where reports, figures, and tables will be written.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(project_root / "models"),
        help="Directory where trained model artifacts will be saved.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used across data splitting and modeling.",
    )
    parser.add_argument(
        "--false-positive-cost",
        type=float,
        default=25.0,
        help="Business cost assumption for flagging a valid transaction.",
    )
    parser.add_argument(
        "--false-negative-cost",
        type=float,
        default=500.0,
        help="Business cost assumption for missing a fraudulent transaction.",
    )
    parser.add_argument(
        "--skip-deep-learning",
        action="store_true",
        help="Skip TensorFlow training even if tensorflow is installed.",
    )
    return parser.parse_args()


def require_package(package_available: Any, install_name: str) -> None:
    if package_available is None:
        raise ImportError(f"Missing dependency: install `{install_name}` from requirements.txt before running.")


def ensure_training_dependencies(include_deep_learning: bool) -> None:
    require_package(Pipeline, "scikit-learn")
    require_package(SMOTE, "imbalanced-learn")
    if include_deep_learning:
        require_package(keras, "tensorflow")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Place creditcard.csv or creditcard.csv.zip in data/ or pass --input."
        )

    if dataset_path.suffix.lower() == ".zip":
        return pd.read_csv(dataset_path, compression="zip")
    return pd.read_csv(dataset_path)


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    frame["LogAmount"] = np.log1p(frame["Amount"].clip(lower=0))
    frame["Hour"] = ((frame["Time"] // 3600) % 24).astype(int)
    frame["DayIndex"] = (frame["Time"] // 86400).astype(int)
    return frame


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    clean = raw.copy()
    duplicate_rows = int(clean.duplicated().sum())
    missing_values = int(clean.isna().sum().sum())
    clean = clean.drop_duplicates().reset_index(drop=True)
    clean["Class"] = clean["Class"].astype(int)
    clean = engineer_features(clean)

    summary = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "duplicates_removed": duplicate_rows,
        "missing_values": missing_values,
        "fraud_rows": int(clean["Class"].sum()),
        "normal_rows": int((clean["Class"] == 0).sum()),
        "fraud_rate_pct": float(clean["Class"].mean() * 100),
        "avg_amount": float(clean["Amount"].mean()),
        "median_amount": float(clean["Amount"].median()),
        "max_amount": float(clean["Amount"].max()),
    }
    return clean, summary


def split_dataset(clean: pd.DataFrame, random_state: int) -> DatasetBundle:
    features = clean[MODEL_FEATURES].copy()
    target = clean["Class"].copy()

    X_temp, X_test, y_temp, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=random_state,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        stratify=y_temp,
        random_state=random_state,
    )

    return DatasetBundle(
        raw=clean,
        clean=clean,
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def build_logistic_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=None,
                ),
            ),
        ]
    )


def build_random_forest(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def smote_resample(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    sampler = SMOTE(random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0

    precision = precision[:-1]
    recall = recall[:-1]
    scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_index = int(scores.argmax())
    return float(thresholds[best_index]), float(scores[best_index])


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float | int | str]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    total_cost = fp * false_positive_cost + fn * false_negative_cost

    return {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "estimated_cost": float(total_cost),
    }


def plot_visuals(clean: pd.DataFrame, output_dir: Path) -> list[str]:
    if plt is None or sns is None:
        return ["Skipped chart generation because matplotlib/seaborn are not installed."]

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    fraud_share = clean["Class"].value_counts().rename(index={0: "Normal", 1: "Fraud"})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=fraud_share.index,
        y=fraud_share.values,
        hue=fraud_share.index,
        palette=["#1f4e79", "#c44536"],
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Fraud Distribution")
    ax.set_ylabel("Transactions")
    fig.tight_layout()
    fraud_distribution_path = figures_dir / "fraud_distribution.png"
    fig.savefig(fraud_distribution_path, dpi=160)
    plt.close(fig)
    created.append("figures/fraud_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=clean, x="Class", y="Amount", hue="Class", palette=["#1f4e79", "#c44536"], legend=False, ax=ax)
    ax.set_title("Transaction Amount by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Amount")
    fig.tight_layout()
    amount_path = figures_dir / "amount_by_class.png"
    fig.savefig(amount_path, dpi=160)
    plt.close(fig)
    created.append("figures/amount_by_class.png")

    hourly = clean.groupby(["Hour", "Class"]).size().reset_index(name="Transactions")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=hourly, x="Hour", y="Transactions", hue="Class", palette=["#1f4e79", "#c44536"], ax=ax)
    ax.set_title("Hourly Transaction Pattern")
    ax.set_ylabel("Transactions")
    fig.tight_layout()
    hourly_path = figures_dir / "hourly_pattern.png"
    fig.savefig(hourly_path, dpi=160)
    plt.close(fig)
    created.append("figures/hourly_pattern.png")

    corr = clean[["Amount", "Time", "LogAmount", "Hour", "Class", *[f"V{index}" for index in range(1, 9)]]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Snapshot")
    fig.tight_layout()
    heatmap_path = figures_dir / "correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=160)
    plt.close(fig)
    created.append("figures/correlation_heatmap.png")

    return created


def train_baseline_models(
    bundle: DatasetBundle,
    random_state: int,
    false_positive_cost: float,
    false_negative_cost: float,
) -> tuple[list[dict[str, float | int | str]], dict[str, Any]]:
    logistic_pipeline = build_logistic_pipeline(random_state)
    X_train_smote, y_train_smote = smote_resample(bundle.X_train, bundle.y_train, random_state)
    logistic_pipeline.fit(X_train_smote, y_train_smote)

    valid_probs_logistic = logistic_pipeline.predict_proba(bundle.X_valid)[:, 1]
    logistic_threshold, _ = find_best_threshold(bundle.y_valid, valid_probs_logistic)
    test_probs_logistic = logistic_pipeline.predict_proba(bundle.X_test)[:, 1]
    logistic_metrics = evaluate_predictions(
        "Logistic Regression + SMOTE",
        bundle.y_test,
        test_probs_logistic,
        logistic_threshold,
        false_positive_cost,
        false_negative_cost,
    )

    rf_model = build_random_forest(random_state)
    rf_model.fit(bundle.X_train, bundle.y_train)
    valid_probs_rf = rf_model.predict_proba(bundle.X_valid)[:, 1]
    rf_threshold, _ = find_best_threshold(bundle.y_valid, valid_probs_rf)
    test_probs_rf = rf_model.predict_proba(bundle.X_test)[:, 1]
    rf_metrics = evaluate_predictions(
        "Random Forest",
        bundle.y_test,
        test_probs_rf,
        rf_threshold,
        false_positive_cost,
        false_negative_cost,
    )

    artifacts = {
        "logistic_pipeline": logistic_pipeline,
        "random_forest_model": rf_model,
        "best_baseline_name": min(
            [logistic_metrics, rf_metrics],
            key=lambda item: (item["estimated_cost"], -item["pr_auc"]),
        )["model"],
        "thresholds": {
            "Logistic Regression + SMOTE": logistic_threshold,
            "Random Forest": rf_threshold,
        },
    }
    return [logistic_metrics, rf_metrics], artifacts


def build_deep_learning_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_neural_network(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.30),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def train_deep_learning_model(
    bundle: DatasetBundle,
    random_state: int,
    false_positive_cost: float,
    false_negative_cost: float,
) -> tuple[dict[str, float | int | str], dict[str, Any]]:
    tf.keras.utils.set_random_seed(random_state)
    preprocessor = build_deep_learning_preprocessor()
    X_train_smote, y_train_smote = smote_resample(bundle.X_train, bundle.y_train, random_state)
    X_train_processed = preprocessor.fit_transform(X_train_smote)
    X_valid_processed = preprocessor.transform(bundle.X_valid)
    X_test_processed = preprocessor.transform(bundle.X_test)

    class_weight = {0: 1.0, 1: max(1.0, (len(bundle.y_train) - bundle.y_train.sum()) / max(bundle.y_train.sum(), 1))}

    model = build_neural_network(X_train_processed.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        )
    ]
    model.fit(
        X_train_processed,
        y_train_smote,
        validation_data=(X_valid_processed, bundle.y_valid),
        epochs=40,
        batch_size=1024,
        verbose=0,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    valid_probs = model.predict(X_valid_processed, verbose=0).ravel()
    threshold, _ = find_best_threshold(bundle.y_valid, valid_probs)
    test_probs = model.predict(X_test_processed, verbose=0).ravel()
    metrics = evaluate_predictions(
        "Deep Neural Network",
        bundle.y_test,
        test_probs,
        threshold,
        false_positive_cost,
        false_negative_cost,
    )

    artifacts = {
        "deep_learning_preprocessor": preprocessor,
        "deep_learning_model": model,
        "threshold": threshold,
    }
    return metrics, artifacts


def build_feature_summary(clean: pd.DataFrame) -> pd.DataFrame:
    feature_frame = clean[MODEL_FEATURES + ["Class"]]
    summary = feature_frame.describe().T
    summary["missing"] = feature_frame.isna().sum()
    summary["fraud_correlation"] = feature_frame.corr(numeric_only=True)["Class"]
    return summary.reset_index().rename(columns={"index": "feature"})


def build_business_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    best_row = metrics_df.sort_values(["estimated_cost", "pr_auc"], ascending=[True, False]).iloc[0]
    return pd.DataFrame(
        [
            {
                "recommended_model": best_row["model"],
                "reason": "Lowest estimated fraud operations cost with strong PR-AUC performance",
                "estimated_cost": best_row["estimated_cost"],
                "precision": best_row["precision"],
                "recall": best_row["recall"],
                "false_positive": best_row["false_positive"],
                "false_negative": best_row["false_negative"],
            }
        ]
    )


def write_markdown_summary(
    quality_summary: dict[str, float | int],
    metrics_df: pd.DataFrame,
    business_summary: pd.DataFrame,
    chart_notes: list[str],
    output_dir: Path,
) -> None:
    best_model = business_summary.iloc[0]
    lines = [
        "# Credit Card Fraud Detection Executive Summary",
        "",
        "## Data Quality",
        f"- Raw transactions: {quality_summary['raw_rows']:,}",
        f"- Clean transactions: {quality_summary['clean_rows']:,}",
        f"- Duplicates removed: {quality_summary['duplicates_removed']:,}",
        f"- Fraud rate: {quality_summary['fraud_rate_pct']:.4f}%",
        "",
        "## Model Recommendation",
        f"- Recommended model: {best_model['recommended_model']}",
        f"- Estimated cost: ${best_model['estimated_cost']:,.2f}",
        f"- Precision: {best_model['precision']:.4f}",
        f"- Recall: {best_model['recall']:.4f}",
        "",
        "## Business Takeaways",
        "- Missing fraud cases is much more expensive than manually reviewing a few extra flagged transactions.",
        "- Threshold tuning matters more than plain accuracy in highly imbalanced fraud detection.",
        "- A production workflow should monitor precision drift and retrain when cardholder behavior shifts.",
        "",
        "## Visualization Notes",
        *[f"- {note}" for note in chart_notes],
        "",
        "## Model Comparison",
        "```text",
        metrics_df.round(4).to_string(index=False),
        "```",
        "",
    ]
    (output_dir / "executive_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_html_report(
    quality_summary: dict[str, float | int],
    metrics_df: pd.DataFrame,
    business_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    best_model = business_summary.iloc[0]
    figure_paths = [
        ("Fraud Distribution", output_dir / "figures" / "fraud_distribution.png"),
        ("Amount by Class", output_dir / "figures" / "amount_by_class.png"),
        ("Hourly Pattern", output_dir / "figures" / "hourly_pattern.png"),
        ("Correlation Heatmap", output_dir / "figures" / "correlation_heatmap.png"),
    ]
    figures_html = []
    for label, path in figure_paths:
        if path.exists():
            figures_html.append(
                f'<div class="figure-card"><h3>{label}</h3><img src="figures/{path.name}" alt="{label}" /></div>'
            )

    top_correlations = (
        feature_summary.loc[feature_summary["feature"] != "Class", ["feature", "fraud_correlation"]]
        .assign(abs_corr=lambda frame: frame["fraud_correlation"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(8)[["feature", "fraud_correlation"]]
    )
    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Credit Card Fraud Detection Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 32px; color: #17202a; background: #f7f9fc; }}
        h1, h2 {{ color: #173f5f; }}
        h3 {{ color: #24577a; margin-bottom: 10px; }}
        .card {{ background: white; padding: 20px; margin-bottom: 18px; border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }}
        .kpi {{ display: inline-block; min-width: 180px; margin-right: 16px; }}
        .grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 18px; }}
        .figure-card {{ background: white; padding: 16px; border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }}
        .figure-card img {{ width: 100%; border-radius: 8px; border: 1px solid #d8dee9; }}
        table {{ border-collapse: collapse; width: 100%; background: white; }}
        th, td {{ border: 1px solid #d8dee9; padding: 10px; text-align: left; }}
        th {{ background: #e9eef6; }}
        ul {{ margin: 0; padding-left: 18px; }}
      </style>
    </head>
    <body>
      <h1>Credit Card Fraud Detection Using Deep Learning</h1>
      <div class="card">
        <h2>Data Quality</h2>
        <div class="kpi"><strong>Raw Rows</strong><br>{quality_summary["raw_rows"]:,}</div>
        <div class="kpi"><strong>Clean Rows</strong><br>{quality_summary["clean_rows"]:,}</div>
        <div class="kpi"><strong>Fraud Rate</strong><br>{quality_summary["fraud_rate_pct"]:.4f}%</div>
        <div class="kpi"><strong>Duplicates Removed</strong><br>{quality_summary["duplicates_removed"]:,}</div>
      </div>
      <div class="card">
        <h2>Recommended Model</h2>
        <p><strong>{best_model["recommended_model"]}</strong> delivers the lowest estimated operations cost.</p>
        <p>Precision: {best_model["precision"]:.4f} | Recall: {best_model["recall"]:.4f} | Estimated Cost: ${best_model["estimated_cost"]:,.2f}</p>
      </div>
      <div class="card">
        <h2>Business Interpretation</h2>
        <ul>
          <li>Fraud is extremely rare, so accuracy alone can look strong even when the model misses costly fraud events.</li>
          <li>Threshold tuning is essential because false negatives are assumed to cost far more than false positives.</li>
          <li>The recommended model balances recall and review workload better than the alternatives in this run.</li>
        </ul>
      </div>
      <div class="card">
        <h2>Model Comparison</h2>
        {metrics_df.to_html(index=False)}
      </div>
      <div class="card">
        <h2>Top Fraud-Linked Features</h2>
        {top_correlations.round(4).to_html(index=False)}
      </div>
      <div class="card">
        <h2>EDA Visuals</h2>
        <div class="grid">
          {''.join(figures_html) if figures_html else '<p>Figures were not generated for this run.</p>'}
        </div>
      </div>
    </body>
    </html>
    """
    (output_dir / "analysis_report.html").write_text(html.strip(), encoding="utf-8")


def save_artifacts(
    baseline_artifacts: dict[str, Any],
    deep_learning_artifacts: dict[str, Any] | None,
    models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    baseline_bundle = {
        "logistic_pipeline": baseline_artifacts["logistic_pipeline"],
        "random_forest_model": baseline_artifacts["random_forest_model"],
        "thresholds": baseline_artifacts["thresholds"],
        "best_baseline_name": baseline_artifacts["best_baseline_name"],
        "feature_columns": MODEL_FEATURES,
    }
    with (models_dir / "baseline_models.pkl").open("wb") as file:
        pickle.dump(baseline_bundle, file)

    if deep_learning_artifacts is not None:
        with (models_dir / "deep_learning_preprocessor.pkl").open("wb") as file:
            pickle.dump(
                {
                    "preprocessor": deep_learning_artifacts["deep_learning_preprocessor"],
                    "threshold": deep_learning_artifacts["threshold"],
                    "feature_columns": MODEL_FEATURES,
                },
                file,
            )
        deep_learning_artifacts["deep_learning_model"].save(models_dir / "fraud_neural_network.keras")


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    include_deep_learning = not args.skip_deep_learning
    ensure_training_dependencies(include_deep_learning=include_deep_learning)

    raw = load_dataset(Path(args.input))
    clean, quality_summary = clean_dataset(raw)
    bundle = split_dataset(clean, args.random_state)

    metrics, baseline_artifacts = train_baseline_models(
        bundle=bundle,
        random_state=args.random_state,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
    )

    deep_learning_artifacts: dict[str, Any] | None = None
    if include_deep_learning:
        deep_learning_metrics, deep_learning_artifacts = train_deep_learning_model(
            bundle=bundle,
            random_state=args.random_state,
            false_positive_cost=args.false_positive_cost,
            false_negative_cost=args.false_negative_cost,
        )
        metrics.append(deep_learning_metrics)

    metrics_df = pd.DataFrame(metrics).sort_values(["estimated_cost", "pr_auc"], ascending=[True, False])
    feature_summary = build_feature_summary(clean)
    business_summary = build_business_summary(metrics_df)
    chart_notes = plot_visuals(clean, output_dir)

    metrics_df.to_csv(output_dir / "model_comparison.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
    pd.DataFrame([quality_summary]).to_csv(output_dir / "data_quality_summary.csv", index=False)
    business_summary.to_csv(output_dir / "business_summary.csv", index=False)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "input_path": str(Path(args.input).resolve()),
                "random_state": args.random_state,
                "false_positive_cost": args.false_positive_cost,
                "false_negative_cost": args.false_negative_cost,
                "deep_learning_enabled": include_deep_learning,
            },
            file,
            indent=2,
        )

    write_markdown_summary(quality_summary, metrics_df, business_summary, chart_notes, output_dir)
    write_html_report(quality_summary, metrics_df, business_summary, feature_summary, output_dir)
    save_artifacts(baseline_artifacts, deep_learning_artifacts, models_dir)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
