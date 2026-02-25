# Genomic Transition Zone Predictor

Predicts the start positions of genomic transition zones from raw nucleotide sequences using machine-learning models trained with [AutoGluon](https://auto.gluon.ai/).

## Transition Zones

A transition zone marks the boundary where one functional region of DNA ends and another begins. This project detects four types:

| Zone | Name | Description | Window size |
|------|------|-------------|-------------|
| **EI** | Exon → Intron | Splice-donor site at the end of an exon | 12 nt |
| **IE** | Intron → Exon | Splice-acceptor site at the start of the next exon | 105 nt |
| **ZE** | Intergenic → First Exon | Promoter / transcription start region | 550 nt |
| **EZ** | Last Exon → Intergenic | Transcription termination region | 550 nt |

## Project Structure

```
.
├── data_ensembl/                  # Raw gene data downloaded from Ensembl
│   ├── 3-187668812-187670494.txt
│   └── 20-1-64444167.txt
│
├── data_extraction/               # Pipeline: raw data → labeled CSVs
│   ├── classes/
│   │   ├── base.py                # TransitionSpec & TransitionExtractor base class
│   │   ├── ei_extracion.py        # EI extractor (window ±5 around intron start)
│   │   ├── ie_extracion.py        # IE extractor (100 left + 5 right of exon start)
│   │   ├── ze_extracion.py        # ZE extractor (500 left + 50 right of first exon)
│   │   └── ez_extracion.py        # EZ extractor (50 left + 500 right of last exon)
│   ├── extract_data.py            # CLI script that reads Ensembl files and writes CSVs
│   ├── principal_extraction_data.ipynb
│   └── README.md                  # Detailed extraction logic and data format
│
├── data/                          # Extracted labeled datasets (output of extraction)
│   ├── data_ei.csv                # 22 391 samples, 12 features (B1–B12) + label
│   ├── data_ie.csv                # 22 391 samples, 105 features (B1–B105) + label
│   ├── data_ze.csv                #  2 606 samples, 550 features (B1–B550) + label
│   └── data_ez.csv                #  2 606 samples, 550 features (B1–B550) + label
│
├── training/                      # Model training & evaluation
│   ├── notebooks/
│   │   ├── 01_ei_autogluon.ipynb  # Train EI model
│   │   ├── 02_ie_autogluon.ipynb  # Train IE model
│   │   ├── 03_ze_autogluon.ipynb  # Train ZE model
│   │   ├── 04_ez_autogluon.ipynb  # Train EZ model
│   │   ├── 05_inference_smoke_test.ipynb  # Validate model loading & predictions
│   │   └── 06_confusion_matrices.ipynb    # Generate single-class confusion matrices
│   ├── models/
│   │   ├── autogluon_ei/          # Saved AutoGluon predictor for EI
│   │   ├── autogluon_ie/          # Saved AutoGluon predictor for IE
│   │   ├── autogluon_ze/          # Saved AutoGluon predictor for ZE
│   │   └── autogluon_ez/          # Saved AutoGluon predictor for EZ
│   ├── reports/
│   │   ├── baseline_metrics.csv           # Per-transition evaluation metrics
│   │   ├── model_registry.json            # Model paths & best metrics (used by API)
│   │   ├── inference_smoke_test.csv       # Smoke-test results
│   │   ├── confusion_matrices.png         # 2×2 grid of confusion matrices
│   │   └── confusion_matrix_summary.csv   # TP/FN/FP/TN summary table
│   └── src/
│       ├── print_baseline_metrics.py      # Saves evaluation metrics to CSV
│       ├── model_selection.py             # Builds model_registry.json from metrics
│       └── generate_confusion_matrices.py # Generates confusion matrices (headless)
│
├── app/                           # FastAPI inference API + web UI
│   ├── main.py                    # Endpoints: /, /api, /health, /predict
│   ├── inference.py               # TransitionInferenceService (sliding-window prediction)
│   ├── schemas.py                 # Pydantic request/response models
│   └── static/
│       └── index.html             # Web interface (single-page, no build step)
│
├── LICENSE                        # MIT License
└── README.MD                      # This file
```

## Pipeline Overview

```
  data_ensembl/*.txt
        │
        ▼
  data_extraction/extract_data.py     ──►  data/data_{ei,ie,ze,ez}.csv
        │
        ▼
  training/notebooks/01–04_*.ipynb    ──►  training/models/autogluon_*
        │
        ▼
  training/src/model_selection.py     ──►  training/reports/model_registry.json
        │
        ▼
  app/main.py  (FastAPI)              ──►  POST /predict  →  JSON response
```

### 1. Data extraction

Raw Ensembl gene files are parsed to identify positive transition positions from exon coordinates. For each positive, a negative sample is generated at a random nearby position (controlled distance) to create balanced binary datasets.

```bash
cd data_extraction
python extract_data.py \
    --input-dir ../data_ensembl \
    --output-dir ../data \
    --negatives-per-positive 1 \
    --random-seed 42
```

### 2. Model training

Each notebook (`01`–`04`) trains a binary AutoGluon classifier for one transition type. Data is split by gene using `GroupShuffleSplit` (60% train / 20% validation / 20% test) to prevent data leakage between genes.

### 3. Model registry

After training, run the model selection script from the project root to build the registry JSON used by the API:

```bash
python training/src/model_selection.py
```

### 4. Confusion matrices

Generate the confusion matrix plot and summary CSV:

```bash
python training/src/generate_confusion_matrices.py
```

This produces `training/reports/confusion_matrices.png` and `training/reports/confusion_matrix_summary.csv`.

## Results

### Baseline Metrics (test set)

| Transition | F1 | Precision | Recall | ROC AUC | Accuracy | MCC |
|------------|------|-----------|--------|---------|----------|------|
| **EI** | 0.984 | 0.995 | 0.973 | 0.998 | 0.985 | 0.970 |
| **IE** | 0.975 | 0.987 | 0.963 | 0.998 | 0.976 | 0.952 |
| **ZE** | 0.610 | 0.512 | 0.755 | 0.712 | 0.617 | 0.279 |
| **EZ** | 0.580 | 0.416 | 0.958 | 0.557 | 0.450 | 0.128 |

### Confusion Matrices

| Transition | TP | FN | FP | TN |
|------------|------|------|------|------|
| **EI** | 2 518 | 70 | 12 | 2 788 |
| **IE** | 2 492 | 96 | 34 | 2 766 |
| **ZE** | 197 | 64 | 188 | 209 |
| **EZ** | 250 | 11 | 351 | 46 |

EI and IE models achieve high accuracy with very few errors. ZE and EZ models have lower precision due to the smaller training set and the larger, more complex nucleotide window (550 nt).

## API & Web Interface

### Requirements

```bash
pip install fastapi uvicorn autogluon.tabular pandas
```

### Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser to use the **web interface**, or `http://localhost:8000/docs` for the Swagger interactive docs.

The web UI allows you to:
- Paste or load a sample nucleotide sequence
- Configure **Top K** (number of hits per zone) and **Min Probability** threshold
- View results as cards for each transition zone with probability bars
- See the sequence with **color-highlighted** regions where transitions were detected

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/api` | API info and configured transitions (JSON) |
| `GET` | `/health` | Model status and resolved paths |
| `POST` | `/predict` | Predict transition zones in a nucleotide sequence |

### `POST /predict`

**Request body:**

```json
{
  "sequence": "acgtacgtacgtacgt...",
  "top_k": 3,
  "min_probability": 0.5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sequence` | string | *required* | Nucleotide sequence (A/C/G/T) |
| `top_k` | int | 3 | Number of top hits to return per transition |
| `min_probability` | float | 0.0 | Minimum probability threshold |

**Response:**

```json
{
  "sequence_length": 1000,
  "transitions": {
    "EI": {
      "window_size": 12,
      "total_windows": 989,
      "returned_hits": 3,
      "hits": [
        {
          "start_index_based": 142,
          "end_index_based": 153,
          "probability": 0.9987
        }
      ]
    },
    "IE": { "..." },
    "ZE": { "..." },
    "EZ": { "..." }
  }
}
```

### Example with `curl`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "acgtacgtacgtacgt", "top_k": 3, "min_probability": 0.5}'
```

## License

MIT License - Carlos Alfredo Serrano Molina
