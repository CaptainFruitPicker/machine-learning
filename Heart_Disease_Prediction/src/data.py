from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Heart_Disease_Prediction.xlsx"

CANONICAL_COLUMNS = {
    "Age": "age",
    "Sex": "sex",
    "Chest pain type": "chest_pain_type",
    "BP": "bp",
    "Cholesterol": "cholesterol",
    "FBS over 120": "fbs_over_120",
    "EKG results": "ekg_results",
    "Max HR": "max_hr",
    "Exercise angina": "exercise_angina",
    "ST depression": "st_depression",
    "Slope of ST": "slope_of_st",
    "Number of vessels fluro": "number_of_vessels_fluro",
    "Thallium": "thallium",
    "Heart Disease": "heart_disease",
    "Са1": "ca1_extra",
}

LOOKUP_COLUMNS = {
    re.sub(r"[^\w]+", "", source.casefold()): target for source, target in CANONICAL_COLUMNS.items()
}

TARGET_COLUMN = "heart_disease"
EXCLUDED_FEATURE_COLUMNS = {"1", "ca1_extra", "artifact_1"}


def _fallback_column_name(name: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "column"


def _canonical_column_name(name: str) -> str:
    original = str(name).strip()
    lookup_key = re.sub(r"[^\w]+", "", original.casefold())
    canonical = LOOKUP_COLUMNS.get(lookup_key, _fallback_column_name(original))

    if re.fullmatch(r"\d+", canonical):
        return f"artifact_{canonical}"

    return canonical


def read_dataframe(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    suffix = data_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path)
    elif suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file type: {data_path.suffix}")

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df


def prepare_dataset(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    df = read_dataframe(path).copy()
    rename_map = {column: _canonical_column_name(str(column)) for column in df.columns}
    df = df.rename(columns=rename_map)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    target = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"presence": 1, "absence": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    )

    if target.isna().any():
        numeric_target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        target = target.fillna(numeric_target)

    if target.isna().any():
        raise ValueError("Target column contains unsupported labels.")

    feature_columns = [
        column for column in df.columns if column != TARGET_COLUMN and column not in EXCLUDED_FEATURE_COLUMNS
    ]
    features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True))
    features = features.fillna(0.0)

    x = features.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    return x, y, feature_columns, df


def stratified_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_value in np.unique(y):
        class_indices = np.where(y == class_value)[0]
        shuffled = rng.permutation(class_indices)

        if len(shuffled) <= 1:
            split_at = len(shuffled)
        else:
            proposed = int(round(len(shuffled) * test_size))
            proposed = max(1, proposed)
            proposed = min(len(shuffled) - 1, proposed)
            split_at = len(shuffled) - proposed

        train_indices.extend(shuffled[:split_at].tolist())
        test_indices.extend(shuffled[split_at:].tolist())

    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]
