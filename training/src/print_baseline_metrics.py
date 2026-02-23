import pandas as pd
from pathlib import Path
from datetime import datetime


class BaselineMetrics:

    def __init__(self, 
    transition: str,
    perf: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ):
        self.transition = transition
        self.perf = perf
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.out_csv = "../reports/baseline_metrics.csv"


    def save_metrics(self):

        perf_clean = {k: float(v) for k, v in self.perf.items()}

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "transition": self.transition,
            "n_train": len(self.train_df),
            "n_val": len(self.val_df),
            "n_test": len(self.test_df),
            **perf_clean
        }

        out_path = Path(self.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        new_row_df = pd.DataFrame([row])
    
        if out_path.exists():
            summary = pd.read_csv(out_path)
            # Deja solo la corrida más reciente por transición:
            summary = summary[summary["transition"] != self.transition]
            summary = pd.concat([summary, new_row_df], ignore_index=True)
        else:
            summary = new_row_df

        # Orden recomendado
        cols_order = [
            "transition", "f1", "precision", "recall", "roc_auc",
            "accuracy", "balanced_accuracy", "mcc",
            "n_train", "n_val", "n_test", "timestamp"
        ]
        existing_cols = [c for c in cols_order if c in summary.columns]
        other_cols = [c for c in summary.columns if c not in existing_cols]
        summary = summary[existing_cols + other_cols].sort_values("transition")
    
        summary.to_csv(out_path, index=False)
