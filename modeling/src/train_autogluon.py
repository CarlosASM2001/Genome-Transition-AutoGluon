import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor


DEFAULT_DROP_COLUMNS = [
    "gene_id",
    "chromosome",
    "global_position",
    "local_position",
]


def parse_drop_columns(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return []
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def split_train_test(
    df: pd.DataFrame, label_column: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    label_counts = df[label_column].value_counts(dropna=False)
    if test_size <= 0:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True), None

    if label_counts.min() < 2:
        # If any class has only one sample, stratified split is not reliable.
        return df.sample(frac=1, random_state=seed).reset_index(drop=True), None

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(label_column):
        shuffled = group.sample(frac=1, random_state=seed)
        test_count = int(round(len(shuffled) * test_size))
        test_count = max(1, test_count)
        if test_count >= len(shuffled):
            test_count = len(shuffled) - 1

        test_parts.append(shuffled.iloc[:test_count])
        train_parts.append(shuffled.iloc[test_count:])

    train_df = (
        pd.concat(train_parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat(test_parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    return train_df, test_df


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Train an AutoGluon model from the merged transition dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=project_root / "modeling" / "data" / "processed" / "transition_dataset.csv",
        help="CSV file produced by build_training_dataset.py",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="transition_label",
        help="Target column name.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "modeling" / "artifacts" / "autogluon_model",
        help="Directory where AutoGluon will save trained models.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=project_root / "modeling" / "outputs",
        help="Directory where metrics and leaderboard files are stored.",
    )
    parser.add_argument(
        "--drop-columns",
        type=str,
        default=",".join(DEFAULT_DROP_COLUMNS),
        help=(
            "Comma-separated feature columns to drop before training. "
            "Useful to avoid leakage from IDs/coordinates."
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=900,
        help="Training budget in seconds.",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="medium_quality",
        help="AutoGluon presets, e.g. medium_quality, high_quality, best_quality.",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="accuracy",
        help="Main evaluation metric for AutoGluon.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    dataset = pd.read_csv(args.dataset_path, low_memory=False)
    if dataset.empty:
        raise ValueError(f"Dataset is empty: {args.dataset_path}")

    if args.label_column not in dataset.columns:
        raise ValueError(
            f"Label column '{args.label_column}' not found in dataset columns."
        )

    columns_to_drop = [
        column
        for column in parse_drop_columns(args.drop_columns)
        if column != args.label_column and column in dataset.columns
    ]
    modeling_df = dataset.drop(columns=columns_to_drop, errors="ignore")

    feature_columns = [col for col in modeling_df.columns if col != args.label_column]
    if not feature_columns:
        raise ValueError("No feature columns available after applying --drop-columns.")
    if modeling_df[args.label_column].nunique(dropna=False) < 2:
        raise ValueError(
            "At least 2 target classes are required to train a classifier."
        )

    train_df, test_df = split_train_test(
        modeling_df,
        label_column=args.label_column,
        test_size=args.test_size,
        seed=args.seed,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label=args.label_column,
        path=str(args.model_dir),
        eval_metric=args.eval_metric,
    )
    predictor.fit(
        train_data=train_df,
        presets=args.presets,
        time_limit=args.time_limit,
    )

    if test_df is not None and not test_df.empty:
        metrics = predictor.evaluate(test_df, silent=True)
        leaderboard = predictor.leaderboard(test_df, silent=True)
        evaluated_on = "test"
        evaluated_rows = len(test_df)
    else:
        metrics = predictor.evaluate(train_df, silent=True)
        leaderboard = predictor.leaderboard(silent=True)
        evaluated_on = "train"
        evaluated_rows = len(train_df)

    leaderboard_path = args.report_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    metrics_payload = {
        "dataset_path": str(args.dataset_path),
        "label_column": args.label_column,
        "dropped_columns": columns_to_drop,
        "train_rows": len(train_df),
        "test_rows": 0 if test_df is None else len(test_df),
        "evaluated_on": evaluated_on,
        "evaluated_rows": evaluated_rows,
        "eval_metric": args.eval_metric,
        "metrics": json_safe(metrics),
    }

    metrics_path = args.report_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=True)

    print(f"Model artifacts saved to: {args.model_dir}")
    print(f"Leaderboard saved to: {leaderboard_path}")
    print(f"Metrics saved to: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
