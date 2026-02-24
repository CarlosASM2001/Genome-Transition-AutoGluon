from fastapi import FastAPI, HTTPException

from app.inference import TRANSITION_WINDOW_SIZES, TransitionInferenceService
from app.schemas import PredictRequest, PredictResponse


app = FastAPI(
    title="Genome Transition Predictor API",
    description=(
        "Predicts probable start positions for genomic transition zones "
        "(EI, IE, ZE, EZ) from a nucleotide sequence."
    ),
    version="0.1.0",
)

_service: TransitionInferenceService | None = None
_service_init_error: Exception | None = None


def get_service() -> TransitionInferenceService:
    global _service, _service_init_error

    if _service is not None:
        return _service
    if _service_init_error is not None:
        raise _service_init_error

    try:
        _service = TransitionInferenceService()
    except Exception as exc:
        _service_init_error = exc
        raise
    return _service


@app.get("/")
def root() -> dict:
    return {
        "message": "Genome Transition Predictor API",
        "transitions": list(TRANSITION_WINDOW_SIZES.keys()),
        "window_sizes": TRANSITION_WINDOW_SIZES,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    try:
        service = get_service()
        return service.health_status()
    except Exception as exc:
        return {
            "status": "error",
            "error": repr(exc),
        }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        service = get_service()
        result = service.predict_all(
            sequence=payload.sequence,
            top_k=payload.top_k,
            min_probability=payload.min_probability,
        )
        return PredictResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
