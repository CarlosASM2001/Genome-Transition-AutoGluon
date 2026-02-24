from typing import Dict, List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sequence: str = Field(..., description="Nucleotide sequence (A/C/G/T).")
    top_k: int = Field(3, ge=1, le=100, description="Top hits to return per transition.")
    min_probability: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum probability required to include a hit.",
    )


class TransitionHit(BaseModel):
    start_index_based: int
    end_index_based: int
    probability: float


class TransitionPrediction(BaseModel):
    window_size: int
    total_windows: int
    returned_hits: int
    hits: List[TransitionHit]


class PredictResponse(BaseModel):
    sequence_length: int
    transitions: Dict[str, TransitionPrediction]