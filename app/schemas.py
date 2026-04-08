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



class PredictorLegacyResponse(BaseModel):
    ei: List[int] = Field(
        ...,
        description="Posiciones start_index_based para transición Exon→Intron (EI).",
    )
    ie: List[int] = Field(
        ...,
        description="Posiciones start_index_based para transición Intron→Exon (IE).",
    )
    ze: List[int] = Field(
        ...,
        description="Posiciones start_index_based para transición Zona intergénica→Exón (ZE).",
    )
    ez: List[int] = Field(
        ...,
        description="Posiciones start_index_based para transición Exón→Zona intergénica (EZ).",
    )