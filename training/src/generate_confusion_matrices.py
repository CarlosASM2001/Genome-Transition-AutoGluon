"""
Genera matrices de confusión single-class para cada zona de transición
(EI, IE, ZE, EZ) usando el conjunto de test y los modelos entrenados.

Salida:
  - training/reports/confusion_matrices.png
  - training/reports/confusion_matrix_summary.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from autogluon.tabular import TabularPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSITION_CSV = {
    "EI": PROJECT_ROOT / "data/data_ei.csv",
    "IE": PROJECT_ROOT / "data/data_ie.csv",
    "ZE": PROJECT_ROOT / "data/data_ze.csv",
    "EZ": PROJECT_ROOT / "data/data_ez.csv",
}

ORDER = ["EI", "IE", "ZE", "EZ"]


def split_by_gene(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    idx_train, idx_test = next(splitter.split(data, groups=data["gene_id"]))
    return data.iloc[idx_train].copy(), data.iloc[idx_test].copy()


def prepare_test_set(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    seq_cols = [c for c in df.columns if c.startswith("B")]
    df_model = df[["gene_id"] + seq_cols + ["label"]].copy()
    df_model["label"] = (
        df_model["label"].astype(str).str.lower().map({"true": 1, "false": 0})
    )
    train_val, test_data = split_by_gene(df_model, test_size=0.2, random_state=42)
    return test_data.drop(columns=["gene_id"])


def normalize_preds(preds) -> np.ndarray:
    s = pd.Series(preds).astype(str).str.strip().str.lower()
    return s.map({"1": 1, "0": 0, "true": 1, "false": 0}).astype(int).values


def main():
    registry_path = PROJECT_ROOT / "training/reports/model_registry.json"
    with registry_path.open() as f:
        registry = json.load(f)

    model_paths = {}
    for t, cfg in registry["models"].items():
        p = Path(cfg["path"])
        model_paths[t] = p if p.is_absolute() else PROJECT_ROOT / p

    results = {}
    for transition in ORDER:
        print(f"\n{'=' * 50}")
        print(f"  {transition}")
        print(f"{'=' * 50}")

        test = prepare_test_set(TRANSITION_CSV[transition])
        X_test = test.drop(columns=["label"])
        y_test = test["label"].values

        predictor = TabularPredictor.load(str(model_paths[transition]), require_py_version_match=False)
        y_pred = normalize_preds(predictor.predict(X_test))

        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        results[transition] = {"cm": cm, "y_test": y_test, "y_pred": y_pred}

        print(f"  Test samples : {len(y_test)}")
        print(f"  TP={cm[0,0]}  FN={cm[0,1]}")
        print(f"  FP={cm[1,0]}  TN={cm[1,1]}")
        print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Confusion Matrices — Transition Zones",
        fontsize=18, fontweight="bold", y=1.01,
    )

    for ax, transition in zip(axes.flat, ORDER):
        cm = results[transition]["cm"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Positive", "Negative"],
            yticklabels=["Positive", "Negative"],
            ax=ax,
            annot_kws={"size": 16, "fontweight": "bold"},
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(f"Confusion Matrix - {transition}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12)

    plt.tight_layout()
    out_png = PROJECT_ROOT / "training/reports/confusion_matrices.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_png}")

    # --- Summary CSV ---
    rows = []
    for transition in ORDER:
        cm = results[transition]["cm"]
        tp, fn = cm[0]
        fp, tn = cm[1]
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({
            "transition": transition,
            "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    summary_df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "training/reports/confusion_matrix_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
