# Genome Transition - AutoGluon

Proyecto para extracción de ventanas genómicas y entrenamiento de modelos de machine learning con AutoGluon.

## Estructura del proyecto

```text
.
├── data/                          # CSV extraídos listos para modelado
│   ├── data_ei.csv
│   ├── data_ie.csv
│   ├── data_ze.csv
│   └── data_ez.csv
├── data_ensembl/                  # Entradas crudas .txt
├── data_extraction/               # Scripts/notebooks de extracción
│   ├── extract_data.py
│   └── principal_extraction_data.ipynb
├── modeling/                      # Pipeline de modelado con AutoGluon
│   ├── artifacts/
│   ├── data/processed/
│   ├── outputs/
│   ├── src/
│   │   ├── build_training_dataset.py
│   │   ├── train_autogluon.py
│   │   └── predict_autogluon.py
│   └── README.md
└── requirements.txt
```

## Inicio rápido

1. Instala dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. Construye el dataset unificado para entrenamiento:

```bash
python modeling/src/build_training_dataset.py
```

3. Entrena AutoGluon:

```bash
python modeling/src/train_autogluon.py --time-limit 900 --presets medium_quality
```

4. Ejecuta predicciones:

```bash
python modeling/src/predict_autogluon.py \
  --input-path modeling/data/processed/transition_dataset.csv \
  --include-proba
```

Para detalles de argumentos y mejores prácticas de modelado, revisa `modeling/README.md`.
