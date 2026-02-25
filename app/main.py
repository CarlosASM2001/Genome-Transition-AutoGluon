from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.inference import TransitionInferenceService, TRANSITION_WINDOW_SIZES
from app.schemas import PredictRequest, PredictResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Genome Transition Predictor API",
    description=(
        "Predicts probable start positions for genomic transition zones "
        "(EI, IE, ZE, EZ) from a nucleotide sequence."
    ),
    version="0.1.0",
)




def get_service() -> TransitionInferenceService:

    try:
        _service = TransitionInferenceService()
    except Exception as exc:
        print(exc)
        raise

    return _service



@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api")
def api_info() -> dict:
    return {
        "message": "Genome Transition Predictor API",
        "transitions": list(TRANSITION_WINDOW_SIZES),
        "windows_size": TRANSITION_WINDOW_SIZES,
        "docs": "/docs",
        "ui": "/",
    }

@app.get("/health")
def health() -> dict:
    try:
        service = get_service()
        return service.health_status()
    except Exception as exc:
        return{
            "status":"error",
            "error": repr(exc)
        }



@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try: 
        service = get_service()
        result = service.predict_all(
            sequence = payload.sequence,
            top_k = payload.top_k,
            min_probability= payload.min_probability
        )
        return PredictResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

 