import json
import heapq
from pathlib import Path
from autogluon.tabular import TabularPredictor
from typing import Dict,List,Iterable,Tuple
import pandas as pd

TRANSITION_WINDOW_SIZES: Dict[str, int] = {
    "EI": 12,
    "IE": 105,
    "ZE": 550,
    "EZ": 550,
}


class TransitionInferenceService:


    def __init__(self, registry_path:Path | None = None , chunk_size: int = 2048):

        self.project_root = Path().resolve()
        self.registry_path =  self.project_root / "training/reports/model_registry.json"
        self.chunck_size = chunk_size
        self.registry_json = self.load_json()
        self.predictor_cache = Dict[str,TabularPredictor] = {}
        
    
    def load_json(self) -> dict:

        with self.registry_path.open("r", encoding="utf-8") as f:
            registry_json = json.load(f)

        return registry_json
    

    def resolve_model_path(self,model_path_value:str) -> Path:

        model_path = Path(model_path_value)
        if model_path.is_absolute():
            return model_path

        return self.project_root / model_path 
        


    def load_predictor(self, transition:str) -> TabularPredictor :
    
        if transition in self.predictor_cache:
            return self.predictor_cache[transition]

        models = self.registry_json["models"].get(transition)

        model_path_value = models.get("path")
        model_path = self.resolve_model_path(model_path_value)

        predictor = TabularPredictor.load(str(model_path))
        self.predictor_cache = predictor

        return predictor
    


    @staticmethod
    def positive_probabilty(probabilities: pd.DataFrame | pd.Series) -> pd.Series:

        if isinstance(probabilities,pd.Series):
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


    def _iter_window_batches(self, sequence: str, window_size: int) -> Iterable[Tuple[pd.DataFrame, List[int]]]:
        total_windows = len(sequence) - window_size + 1
        if total_windows <= 0:
            return

        for batch_start in range(0, total_windows, self.chunk_size):
            batch_end = min(batch_start + self.chunk_size, total_windows)
            starts_1based = [i + 1 for i in range(batch_start, batch_end)]
            windows = [sequence[i : i + window_size] for i in range(batch_start, batch_end)]
            yield self._build_feature_frame(windows, window_size), starts_1based


    def predict_transition(self, transition:str,sequence:str, top_k: int =3, min_probability: float=0.0) -> dict:

        window_size = TRANSITION_WINDOW_SIZES[transition]

        total_windows = max(len(sequence)- window_size + 1,0)

        result = {
            "window_size": window_size,
            "total_windows": total_windows,
            "returned_hits": 0,
            "hits": [],
        }

        if total_windows == 0:
            return result

        predictor = self.load_predictor(transition)
        heap = List[Tuple[float,int]]


        for feature_batch, starts_based in self._iter_window_batches(sequence,window_size):

            probas = predictor.predict_proba(feature_batch, as_pandas=True)
            positive_scores = self.positive_probabilty(probas)

            for start_idx, score in zip(starts_based, positive_scores):
                score_float = float(score)
                if score_float < min_probability:
                    continue

                item = (score_float, start_idx)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        top_items = sorted(heap, key=lambda x : (x[0],x[1]), reverse=True)

        hits = [
            {
                "start_index_based": start_idx,
                "end_index_based": start_idx + window_size - 1,
                "probability": score,
            }
            for score, start_idx in top_items
        ]

        result["returned_hits"] = len(hits)
        result["hits"] = hits

        return result
    

    def predict_all(self, sequence: str, top_k: int = 3, min_probability: float = 0.0) -> dict:
        
        transitions = {
            transition: self.predict_transition(
                transition=transition,
                sequence=sequence,
                top_k=top_k,
                min_probability=min_probability,
            )
            for transition in TRANSITION_WINDOW_SIZES
        }
        return {
            "sequence_length": len(sequence),
            "transitions": transitions,
        }

    def health_status(self) -> dict:
        configured_models = self.registry_json.get("models", {})
        resolved_paths = {}
        missing_paths = []
        for transition, cfg in configured_models.items():
            raw_path = cfg.get("path", "")
            resolved = self.resolve_model_path(raw_path)
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








        


    













