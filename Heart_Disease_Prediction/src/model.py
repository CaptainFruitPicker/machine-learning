from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | dict[str, int]]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(1, len(y_true))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.05
    epochs: int = 6000
    reg_strength: float = 0.01
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.loss_history_: list[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "LogisticRegressionGD":
        samples, features = x.shape
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0

        x_scaled = (x - self.mean_) / self.scale_
        self.weights_ = np.zeros(features, dtype=float)
        self.bias_ = 0.0
        self.feature_names_ = feature_names or [f"feature_{index}" for index in range(features)]
        self.loss_history_ = []

        for epoch in range(self.epochs):
            logits = x_scaled @ self.weights_ + self.bias_
            predictions = sigmoid(logits)

            error = predictions - y
            grad_w = (x_scaled.T @ error) / samples + self.reg_strength * self.weights_
            grad_b = float(np.mean(error))

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

            if epoch % 250 == 0 or epoch == self.epochs - 1:
                loss = self._binary_cross_entropy(y, predictions)
                self.loss_history_.append(round(loss, 6))

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        x_scaled = (x - self.mean_) / self.scale_
        return sigmoid(x_scaled @ self.weights_ + self.bias_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= self.threshold).astype(int)

    def standardized_coefficients(self) -> np.ndarray:
        self._check_is_fitted()
        return self.weights_.copy()

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "reg_strength": self.reg_strength,
            "threshold": self.threshold,
            "weights": self.weights_,
            "bias": self.bias_,
            "mean": self.mean_,
            "scale": self.scale_,
            "feature_names": self.feature_names_,
            "loss_history": self.loss_history_,
            "metadata": metadata or {},
        }

        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> tuple["LogisticRegressionGD", dict]:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)

        model = cls(
            learning_rate=payload["learning_rate"],
            epochs=payload["epochs"],
            reg_strength=payload["reg_strength"],
            threshold=payload["threshold"],
        )
        model.weights_ = payload["weights"]
        model.bias_ = payload["bias"]
        model.mean_ = payload["mean"]
        model.scale_ = payload["scale"]
        model.feature_names_ = payload["feature_names"]
        model.loss_history_ = payload.get("loss_history", [])
        return model, payload.get("metadata", {})

    def export_training_summary(self, path: str | Path, metrics: dict) -> None:
        Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    def _check_is_fitted(self) -> None:
        if self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("Model is not fitted yet.")

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
