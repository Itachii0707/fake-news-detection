from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config
from .data import load_dataset
from .features import build_pipeline
from .utils import set_seed


def train(
    data_path: Path,
    model_path: Path,
    test_size: float,
    random_state: int,
) -> Dict[str, Any]:
    set_seed(random_state)

    X, y = load_dataset(data_path)

    # Encode labels if they are not numeric
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        roc_auc = metrics.roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = np.nan

    report = metrics.classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    cm = metrics.confusion_matrix(y_test, y_pred)

    # Save model bundle (pipeline + label encoder + metadata)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "label_encoder": label_encoder,
        "label_names": list(label_encoder.classes_),
        "config": {
            "test_size": test_size,
            "random_state": random_state,
        },
        "metrics": {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        },
    }
    joblib.dump(bundle, model_path)

    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train a fake news detector (TF-IDF + Logistic Regression)")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.RAW_DATA_PATH,
        help=f"Path to CSV dataset (default: {config.RAW_DATA_PATH})"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=config.MODEL_PATH,
        help=f"Path to save model (default: {config.MODEL_PATH})"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=config.TEST_SIZE,
        help=f"Test size fraction (default: {config.TEST_SIZE})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_STATE,
        help=f"Random seed (default: {config.RANDOM_STATE})"
    )
    args = parser.parse_args()

    bundle = train(
        data_path=args.data,
        model_path=args.model,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print("\n=== Training complete ===")
    print("Model saved to:", args.model)
    print("\n--- Metrics ---")
    print(bundle["metrics"]["classification_report"])
    if bundle["metrics"]["roc_auc"] is not None:
        print(f"ROC-AUC: {bundle['metrics']['roc_auc']:.4f}")
    print("Confusion Matrix:\n", np.array(bundle["metrics"]["confusion_matrix"]))


if __name__ == "__main__":
    main()
