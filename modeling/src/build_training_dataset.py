import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_LABEL_MAPPING = {
    "data_ei.csv": "EI",
    "data_ie.csv": "IE",
    "data_ze.csv": "ZE",
    "data_ez.csv": "EZ",
}

POSITION_COLUMNS = [
    "Intron_Start",
    "Exon_Start",
    "First_Exon_Start",
    "Last_Exon_End",
]

B_COLUMN_REGEX = re.compile(r"^B(\d+)$")


def parse_label_mapping(items: list[str]) -> dict[str, str]:
    if not items:
        return DEFAULT_LABEL_MAPPING.copy()

    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --map value '{item}'. Use the format: file_name.csv=LABEL"
            )
        file_name, label = item.split("=", 1)
        file_name = file_name.strip()
        label = label.strip()
        if not file_name or not label:
            raise ValueError(
                f"Invalid --map value '{item}'. File name and label are required."
            )
        mapping[file_name] = label
    return mapping


def get_b_columns(columns: list[str]) -> list[str]:
    b_columns: list[tuple[int, str]] = []
    for col in columns:
        match = B_COLUMN_REGEX.match(col)
        if match:
            b_columns.append((int(match.group(1)), col))
    return [name for _, name in sorted(b_columns, key=lambda item: item[0])]


def normalize_position_column(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    available = [col for col in POSITION_COLUMNS if col in df.columns]
    if len(available) != 1:
        raise ValueError(
            f"Expected exactly one local position column in '{source_name}', "
            f"found: {available if available else 'none'}."
        )

    return df.rename(columns={available[0]: "local_position"})


def load_labeled_dataframe(csv_path: Path, label: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    df = normalize_position_column(df, csv_path.name)
    b_columns = get_b_columns(df.columns.tolist())
    if not b_columns:
        raise ValueError(f"No B1..Bn columns found in: {csv_path}")

    for col in b_columns:
        df[col] = (
            df[col]
            .fillna("n")
            .astype("string")
            .str.strip()
            .str.lower()
        )

    df["transition_label"] = label
    return df


def build_dataset(input_dir: Path, label_mapping: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for file_name, label in label_mapping.items():
        csv_path = input_dir / file_name
        frame = load_labeled_dataframe(csv_path, label)
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    b_columns = get_b_columns(merged.columns.tolist())
    for col in b_columns:
        merged[col] = (
            merged[col]
            .fillna("n")
            .astype("string")
            .str.strip()
            .str.lower()
        )

    ordered_prefix = [
        "gene_id",
        "chromosome",
        "global_position",
        "local_position",
    ]
    ordered_columns = [col for col in ordered_prefix if col in merged.columns]
    ordered_columns.extend(b_columns)
    ordered_columns.append("transition_label")
    extra_columns = [col for col in merged.columns if col not in ordered_columns]

    return merged[ordered_columns + extra_columns]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Merge EI/IE/ZE/EZ CSV files into a single labeled training dataset "
            "for AutoGluon."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data",
        help="Directory where the source CSV files are located.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "modeling" / "data" / "processed" / "transition_dataset.csv",
        help="Destination CSV path for the merged dataset.",
    )
    parser.add_argument(
        "--map",
        dest="mapping",
        action="append",
        default=[],
        help=(
            "Label mapping in format file_name.csv=LABEL. "
            "Use multiple times. If omitted, defaults to EI/IE/ZE/EZ mapping."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    label_mapping = parse_label_mapping(args.mapping)
    dataset = build_dataset(args.input_dir, label_mapping)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_path, index=False)

    print(f"Merged dataset saved to: {args.output_path}")
    print(f"Total rows: {len(dataset)}")
    print("Class distribution:")
    print(dataset["transition_label"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
