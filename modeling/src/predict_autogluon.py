import argparse
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Generate predictions with a trained AutoGluon model."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input CSV file with feature rows.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "modeling" / "artifacts" / "autogluon_model",
        help="Directory where AutoGluon model artifacts are stored.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "modeling" / "outputs" / "predictions.csv",
        help="Destination CSV with predictions.",
    )
    parser.add_argument(
        "--drop-columns",
        type=str,
        default=",".join(DEFAULT_DROP_COLUMNS),
        help=(
            "Comma-separated columns to drop before prediction. "
            "Should match training preprocessing."
        ),
    )
    parser.add_argument(
        "--include-proba",
        action="store_true",
        help="Include per-class probability columns in the output CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_path}")

    predictor = TabularPredictor.load(str(args.model_dir))
    raw_df = pd.read_csv(args.input_path, low_memory=False)
    if raw_df.empty:
        raise ValueError(f"Input CSV is empty: {args.input_path}")

    columns_to_drop = [
        col for col in parse_drop_columns(args.drop_columns) if col in raw_df.columns
    ]
    feature_df = raw_df.drop(columns=columns_to_drop, errors="ignore")

    if predictor.label in feature_df.columns:
        feature_df = feature_df.drop(columns=[predictor.label])

    predictions = predictor.predict(feature_df)

    output_df = raw_df.copy()
    output_df["prediction"] = predictions.values

    if args.include_proba:
        probabilities = predictor.predict_proba(feature_df)
        for class_name in probabilities.columns:
            output_df[f"proba_{class_name}"] = probabilities[class_name].values

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
