# FastAPI Inference Service

## Endpoints

- `GET /health` - Validates registry/model paths availability.
- `POST /predict` - Returns top-k probable transition positions for EI, IE, ZE, EZ.

## Example request

```json
{
  "sequence": "ACGTACGTACGT",
  "top_k": 3,
  "min_probability": 0.0
}
```

## Run locally

```bash
uvicorn app.main:app --reload
```
