from pathlib import Path
from fastapi.responses import FileResponse

from fastapi import FastAPI, HTTPException
from app.inference import TransitionInferenceService, TRANSITION_WINDOW_SIZES
from app.schemas import PredictRequest, PredictResponse, PredictorLegacyResponse

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



@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api")
def get_info() -> dict:
    return{
        "message" : "Genome Transition Predictor API",
        "message": "Genome Transition Predictor API",
        "transitions": list(TRANSITION_WINDOW_SIZES),
        "windows_size":TRANSITION_WINDOW_SIZES,
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



def _build_legacy_response(service: TransitionInferenceService, payload: PredictRequest) -> PredictorLegacyResponse:
    result = service.predict_all(
        sequence=payload.sequence,
        top_k=payload.top_k,
        min_probability=payload.min_probability,
    )
    transitions = result.get("transitions", {})
    legacy_payload = {
        "ei": [hit["start_index_based"] for hit in transitions.get("EI", {}).get("hits", [])],
        "ie": [hit["start_index_based"] for hit in transitions.get("IE", {}).get("hits", [])],
        "ze": [hit["start_index_based"] for hit in transitions.get("ZE", {}).get("hits", [])],
        "ez": [hit["start_index_based"] for hit in transitions.get("EZ", {}).get("hits", [])],
    }
    return PredictorLegacyResponse(**legacy_payload)


@app.post("/predict", response_model=PredictorLegacyResponse)
def predict(payload: PredictRequest) -> PredictorLegacyResponse:
    """
    Endpoint por defecto compatible con predictor Java legado.
    """
    try:
        service = get_service()
        return _build_legacy_response(service, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post("/predict-rich", response_model=PredictResponse)
def predict_rich(payload: PredictRequest) -> PredictResponse:
    """
    Endpoint con respuesta detallada (estructura completa por transición).
    """
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
        raise HTTPException(status_code=500, detail=f"Rich inference failed: {exc}") from exc


@app.post("/predict-legacy", response_model=PredictorLegacyResponse)
def predict_legacy(payload: PredictRequest) -> PredictorLegacyResponse:
    """
    Endpoint de compatibilidad para el predictor Java legado.
    Devuelve exactamente el formato esperado por predictor/src/clasificador/AutoMLClasificador.java:
    { "ei": [...], "ie": [...], "ze": [...], "ez": [...] }
    """
    try:
        service = get_service()
        return _build_legacy_response(service, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Legacy inference failed: {exc}") from exc