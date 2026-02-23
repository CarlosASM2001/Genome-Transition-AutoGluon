import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "training/reports/baseline_metrics.csv"
REGISTRY_PATH = PROJECT_ROOT / "training/reports/model_registry.json"

# Transition -> trained predictor directory
MODEL_PATH_MAP = {
    "EI": PROJECT_ROOT / "training/models/autogluon_ei",
    "IE": PROJECT_ROOT / "training/models/autogluon_ie",
    "ZE": PROJECT_ROOT / "training/models/autogluon_ze",
    "EZ": PROJECT_ROOT / "training/models/autogluon_ez",
}


def as_project_relative(path: Path) -> str:
    """Return path relative to project root whenever possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_model_registry() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    df = pd.read_csv(METRICS_PATH)
    df["transition"] = df["transition"].astype(str).str.upper().str.strip()

    # Keep best run per transition prioritizing F1
    sort_cols = ["transition", "f1"]
    ascending = [True, False]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
        ascending.append(False)

    best = df.sort_values(sort_cols, ascending=ascending).drop_duplicates(
        "transition", keep="first"
    )

    registry = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_metrics": as_project_relative(METRICS_PATH),
        "metric_priority": "f1",
        "models": {},
    }

    for _, row in best.iterrows():
        transition = row["transition"]
        model_dir = MODEL_PATH_MAP.get(transition)
        if model_dir is None:
            continue

        predictor_file = model_dir / "predictor.pkl"
        if not predictor_file.exists():
            raise FileNotFoundError(f"Missing predictor artifact: {predictor_file}")

        registry["models"][transition] = {
            "path": as_project_relative(model_dir),
            "f1": float(row["f1"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "roc_auc": float(row["roc_auc"]),
            "timestamp": str(row.get("timestamp", "")),
        }

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    return registry


if __name__ == "__main__":
    output = build_model_registry()
    print(f"Registry created at: {REGISTRY_PATH}")
    print(json.dumps(output, indent=2, ensure_ascii=False))
