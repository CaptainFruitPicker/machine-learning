from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import DEFAULT_DATA_PATH, prepare_dataset, stratified_train_test_split
from model import LogisticRegressionGD, classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "outputs" / "evaluation.json"


def _display_path(path: Path) -> str:
    resolved = path.resolve()

    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained heart disease model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the dataset.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained model.")
    parser.add_argument("--output", type=Path, default=DEFAULT_EVAL_PATH, help="Where to save evaluation metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, metadata = LogisticRegressionGD.load(args.model)
    x, y, _, _ = prepare_dataset(args.data)

    seed = int(metadata.get("seed", 42))
    test_size = float(metadata.get("test_size", 0.2))
    _, x_test, _, y_test = stratified_train_test_split(x, y, test_size=test_size, seed=seed)

    predictions = model.predict(x_test)
    metrics = classification_metrics(y_test, predictions)
    metrics.update(
        {
            "evaluation_samples": int(len(x_test)),
            "seed": seed,
            "test_size": test_size,
            "model_path": _display_path(args.model),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Evaluation finished. Accuracy: {metrics['accuracy']}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
