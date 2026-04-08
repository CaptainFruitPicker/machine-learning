from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model import LogisticRegressionGD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
DEFAULT_IMPORTANCE_PATH = PROJECT_ROOT / "outputs" / "feature_importance.csv"
DEFAULT_TEXT_PATH = PROJECT_ROOT / "outputs" / "explanation.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain feature importance for the heart disease model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained model.")
    parser.add_argument(
        "--importance-out",
        type=Path,
        default=DEFAULT_IMPORTANCE_PATH,
        help="Where to save feature importance CSV.",
    )
    parser.add_argument(
        "--text-out",
        type=Path,
        default=DEFAULT_TEXT_PATH,
        help="Where to save the plain-text explanation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, metadata = LogisticRegressionGD.load(args.model)

    coefficients = model.standardized_coefficients()
    importance_df = pd.DataFrame(
        {
            "feature": model.feature_names_,
            "coefficient": coefficients,
            "absolute_importance": abs(coefficients),
            "direction": ["increases_risk" if value > 0 else "decreases_risk" for value in coefficients],
        }
    ).sort_values("absolute_importance", ascending=False)

    args.importance_out.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(args.importance_out, index=False)

    top_positive = importance_df[importance_df["coefficient"] > 0].head(5)
    top_negative = importance_df[importance_df["coefficient"] < 0].head(5)

    lines = [
        "Heart disease model explanation",
        f"Model path: {args.model}",
        f"Training seed: {metadata.get('seed', 42)}",
        "",
        "Top risk-increasing features:",
    ]

    for _, row in top_positive.iterrows():
        lines.append(f"- {row['feature']}: coefficient={row['coefficient']:.4f}")

    lines.append("")
    lines.append("Top risk-decreasing features:")

    for _, row in top_negative.iterrows():
        lines.append(f"- {row['feature']}: coefficient={row['coefficient']:.4f}")

    args.text_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Feature importance saved to: {args.importance_out}")
    print(f"Readable explanation saved to: {args.text_out}")


if __name__ == "__main__":
    main()
