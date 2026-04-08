from __future__ import annotations

import argparse
from pathlib import Path

from data import DEFAULT_DATA_PATH, prepare_dataset, stratified_train_test_split
from model import LogisticRegressionGD, classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "train_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a heart disease prediction model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the dataset.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the trained model.")
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH, help="Where to save training metrics.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Gradient descent learning rate.")
    parser.add_argument("--epochs", type=int, default=6000, help="Training epochs.")
    parser.add_argument("--reg-strength", type=float, default=0.01, help="L2 regularization strength.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y, feature_names, _ = prepare_dataset(args.data)
    x_train, x_test, y_train, y_test = stratified_train_test_split(x, y, test_size=args.test_size, seed=args.seed)

    model = LogisticRegressionGD(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        reg_strength=args.reg_strength,
    )
    model.fit(x_train, y_train, feature_names=feature_names)

    predictions = model.predict(x_test)
    metrics = classification_metrics(y_test, predictions)
    metrics.update(
        {
            "train_samples": int(len(x_train)),
            "validation_samples": int(len(x_test)),
            "seed": args.seed,
            "test_size": args.test_size,
            "loss_checkpoints": model.loss_history_,
        }
    )

    model.save(
        args.model_out,
        metadata={
            "data_path": str(args.data),
            "seed": args.seed,
            "test_size": args.test_size,
        },
    )
    model.export_training_summary(args.metrics_out, metrics)

    print(f"Training finished. Model saved to: {args.model_out}")
    print(f"Validation accuracy: {metrics['accuracy']}")
    print(f"Validation f1: {metrics['f1']}")
    print(f"Metrics saved to: {args.metrics_out}")


if __name__ == "__main__":
    main()
