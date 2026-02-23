import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

metrics_path = Path("training/reports/baseline_metrics.csv")
registry_path = Path("training/reports/model_registry.json")

# Mapa transición -> carpeta del predictor
model_path_map = {
    "EI": "training/models/autogluon_ei",
    "IE": "training/models/autogluon_ie",
    "ZE": "training/models/autogluon_ze",
    "EZ": "training/models/autogluon_ez",
}

df = pd.read_csv(metrics_path)
df["transition"] = df["transition"].astype(str).str.upper().str.strip()

# Si mañana tienes múltiples corridas por transición, toma la mejor por f1
sort_cols = ["transition", "f1"]
ascending = [True, False]
if "timestamp" in df.columns:
    sort_cols.append("timestamp")
    ascending.append(False)

best = df.sort_values(sort_cols, ascending=ascending).drop_duplicates("transition", keep="first")

registry = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_metrics": str(metrics_path),
    "metric_priority": "f1",
    "models": {}
}

for _, row in best.iterrows():
    t = row["transition"]
    if t not in model_path_map:
        continue

    model_dir = Path(model_path_map[t])
    predictor_file = model_dir / "predictor.pkl"
    if not predictor_file.exists():
        raise FileNotFoundError(f"No existe {predictor_file}")

    registry["models"][t] = {
        "path": str(model_dir),
        "f1": float(row["f1"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "roc_auc": float(row["roc_auc"]),
        "timestamp": str(row.get("timestamp", "")),
    }

registry_path.parent.mkdir(parents=True, exist_ok=True)
with registry_path.open("w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2, ensure_ascii=False)

print(f"Registry creado: {registry_path}")
print(json.dumps(registry, indent=2, ensure_ascii=False))
