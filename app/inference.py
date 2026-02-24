import heapq
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor


TRANSITION_WINDOW_SIZES: Dict[str, int] = {
    "EI": 12,
    "IE": 105,
    "ZE": 550,
    "EZ": 550,
}


class TransitionInferenceService:
    def __init__(self, registry_path: Path | None = None, chunk_size: int = 2048):
        self.project_root = Path(__file__).resolve().parents[1]
        self.registry_path = (
            Path(registry_path)
            if registry_path is not None
            else self.project_root / "training/reports/model_registry.json"
        )
        self.chunk_size = chunk_size
        self._registry = self._load_registry()
        self._predictor_cache: Dict[str, TabularPredictor] = {}

    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry not found at: {self.registry_path}. "
                "Generate it with training/src/model_selection.py."
            )
        with self.registry_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
        if "models" not in registry or not isinstance(registry["models"], dict):
            raise ValueError("Invalid model_registry.json format: missing 'models' mapping.")
        return registry

    def _resolve_model_path(self, model_path_value: str) -> Path:
        model_path = Path(model_path_value)
        if model_path.is_absolute():
            return model_path
        return self.project_root / model_path

    def _get_predictor(self, transition: str) -> TabularPredictor:
        if transition in self._predictor_cache:
            return self._predictor_cache[transition]

        model_cfg = self._registry["models"].get(transition)
        if model_cfg is None:
            raise ValueError(f"Transition '{transition}' is not configured in model_registry.json.")

        model_path_value = model_cfg.get("path")
        if not model_path_value:
            raise ValueError(f"Missing 'path' in model registry for transition '{transition}'.")

        model_path = self._resolve_model_path(model_path_value)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist for transition '{transition}': {model_path}"
            )

        predictor = TabularPredictor.load(str(model_path))
        self._predictor_cache[transition] = predictor
        return predictor

    @staticmethod
    def normalize_sequence(raw_sequence: str) -> str:
        normalized = "".join(raw_sequence.split()).lower()
        if not normalized:
            raise ValueError("Sequence is empty after removing whitespace.")
        invalid_chars = sorted({ch for ch in normalized if ch not in {"a", "c", "g", "t"}})
        if invalid_chars:
            raise ValueError(
                "Sequence contains invalid characters. "
                f"Only A/C/G/T are allowed. Found: {''.join(invalid_chars)}"
            )
        return normalized

    @staticmethod
    def _positive_probability(probabilities: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(probabilities, pd.Series):
            return probabilities.astype(float).reset_index(drop=True)

        candidate_columns = [1, "1", True, "True", "true", "positive", "pos", "label_1"]
        for col in candidate_columns:
            if col in probabilities.columns:
                return probabilities[col].astype(float).reset_index(drop=True)

        if probabilities.shape[1] == 2:
            return probabilities.iloc[:, 1].astype(float).reset_index(drop=True)

        return probabilities.max(axis=1).astype(float).reset_index(drop=True)

    @staticmethod
    def _build_feature_frame(sequence_windows: List[str], window_size: int) -> pd.DataFrame:
        columns = [f"B{i}" for i in range(1, window_size + 1)]
        rows = [list(window) for window in sequence_windows]
        return pd.DataFrame(rows, columns=columns)

    def _iter_window_batches(
        self, sequence: str, window_size: int
    ) -> Iterable[Tuple[pd.DataFrame, List[int]]]:
        total_windows = len(sequence) - window_size + 1
        if total_windows <= 0:
            return

        for batch_start in range(0, total_windows, self.chunk_size):
            batch_end = min(batch_start + self.chunk_size, total_windows)
            starts_1based = [i + 1 for i in range(batch_start, batch_end)]
            windows = [sequence[i : i + window_size] for i in range(batch_start, batch_end)]
            yield self._build_feature_frame(windows, window_size), starts_1based

    def predict_transition(
        self, transition: str, sequence: str, top_k: int = 3, min_probability: float = 0.0
    ) -> dict:
        if transition not in TRANSITION_WINDOW_SIZES:
            raise ValueError(f"Unsupported transition '{transition}'.")

        window_size = TRANSITION_WINDOW_SIZES[transition]
        total_windows = max(len(sequence) - window_size + 1, 0)
        result = {
            "window_size": window_size,
            "total_windows": total_windows,
            "returned_hits": 0,
            "hits": [],
        }
        if total_windows == 0:
            return result

        predictor = self._get_predictor(transition)
        heap: List[Tuple[float, int]] = []

        for feature_batch, starts_1based in self._iter_window_batches(sequence, window_size):
            probas = predictor.predict_proba(feature_batch, as_pandas=True)
            positive_scores = self._positive_probability(probas)

            for start_idx, score in zip(starts_1based, positive_scores):
                score_float = float(score)
                if score_float < min_probability:
                    continue

                item = (score_float, start_idx)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        top_items = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)
        hits = [
            {
                "start_index_1based": start_idx,
                "end_index_1based": start_idx + window_size - 1,
                "probability": score,
            }
            for score, start_idx in top_items
        ]
        result["returned_hits"] = len(hits)
        result["hits"] = hits
        return result

    def predict_all(self, sequence: str, top_k: int = 3, min_probability: float = 0.0) -> dict:
        normalized_sequence = self.normalize_sequence(sequence)
        transitions = {
            transition: self.predict_transition(
                transition=transition,
                sequence=normalized_sequence,
                top_k=top_k,
                min_probability=min_probability,
            )
            for transition in TRANSITION_WINDOW_SIZES
        }
        return {
            "sequence_length": len(normalized_sequence),
            "transitions": transitions,
        }

    def health_status(self) -> dict:
        configured_models = self._registry.get("models", {})
        resolved_paths = {}
        missing_paths = []
        for transition, cfg in configured_models.items():
            raw_path = cfg.get("path", "")
            resolved = self._resolve_model_path(raw_path)
            resolved_paths[transition] = str(resolved)
            if not resolved.exists():
                missing_paths.append(transition)

        return {
            "status": "ok" if not missing_paths else "degraded",
            "registry_path": str(self.registry_path),
            "configured_transitions": sorted(configured_models.keys()),
            "resolved_model_paths": resolved_paths,
            "missing_model_paths": sorted(missing_paths),
        }
