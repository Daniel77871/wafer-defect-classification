"""SVM baseline model helpers."""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.load import PROJECT_ROOT
from src.utils.io import ensure_dir


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> SVC:
    """Train a standardized linear one-vs-one SVM and save scaler plus model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = SVC(kernel="linear", decision_function_shape="ovo")
    model.fit(X_train_scaled, y_train)
    val_predictions = model.predict(X_val_scaled)
    val_accuracy = float(accuracy_score(y_val, val_predictions))

    output_dir = ensure_dir(PROJECT_ROOT / "results/exp_F_svm")
    with (output_dir / "svm_model.pkl").open("wb") as file:
        pickle.dump(
            {
                "scaler": scaler,
                "model": model,
                "val_accuracy": val_accuracy,
            },
            file,
        )

    return model
