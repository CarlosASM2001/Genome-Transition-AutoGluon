# Genomic Transition Zone Extractor

## Project Objective

The objective of this project is to extract specific genomic transition zones from raw nucleotide strings. These transition zones include:

* EI – Exon → Intron transition
* IE – Intron → Exon transition
* ZE – Intergenic zone → First Exon transition
* EZ – Last Exon → Intergenic zone transition

The extraction of examples of transition zones from data collected from Ensembl is performed with the Jupyter notebook located in data_extraction/. The extracted transitions are saved as CSV files in data/ and subsequently used to train several machine‑learning models.

As a final product, the project exposes a FastAPI endpoint that returns the probable start positions of each transition zone for any nucleotide sequence provided by the user. This capability has as propose to support genomic analysis for biological and medical research.

## API quick start

The inference API uses models listed in `training/reports/model_registry.json`.

Run the server:

```bash
uvicorn app.main:app --reload
```

Main endpoints:

- `GET /health`
- `POST /predict`