# Modelado con AutoGluon (Tabular)

Este módulo toma los CSV generados por la extracción (`data/data_ei.csv`, `data/data_ie.csv`, `data/data_ze.csv`, `data/data_ez.csv`) y entrena un modelo con AutoGluon.

## Estructura recomendada

```text
modeling/
├── artifacts/                 # Modelos entrenados (AutoGluon)
├── data/
│   └── processed/             # Dataset unificado para entrenamiento
├── outputs/                   # Métricas, leaderboard, predicciones
├── src/
│   ├── build_training_dataset.py
│   ├── train_autogluon.py
│   └── predict_autogluon.py
└── README.md
```

## 1) Instalar dependencias

Desde la raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Construir dataset de entrenamiento

Unifica los 4 CSV en un solo archivo etiquetado (`transition_label`).

```bash
python modeling/src/build_training_dataset.py
```

Salida por defecto:

- `modeling/data/processed/transition_dataset.csv`

Mapeo por defecto:

- `data_ei.csv -> EI`
- `data_ie.csv -> IE`
- `data_ze.csv -> ZE`
- `data_ez.csv -> EZ`

Si quieres usar etiquetas personalizadas:

```bash
python modeling/src/build_training_dataset.py \
  --map data_ei.csv=EI \
  --map data_ie.csv=IE \
  --map data_ze.csv=ZE \
  --map data_ez.csv=EZ
```

## 3) Entrenar modelo con AutoGluon

```bash
python modeling/src/train_autogluon.py \
  --time-limit 900 \
  --presets medium_quality \
  --eval-metric accuracy
```

Salida por defecto:

- Modelo: `modeling/artifacts/autogluon_model/`
- Leaderboard: `modeling/outputs/leaderboard.csv`
- Métricas: `modeling/outputs/metrics.json`

### Notas importantes

- El script elimina por defecto columnas potencialmente sensibles a fuga de información:
  - `gene_id`, `chromosome`, `global_position`, `local_position`
- La variable objetivo por defecto es:
  - `transition_label`

## 4) Generar predicciones

Ejemplo sobre el mismo dataset unificado:

```bash
python modeling/src/predict_autogluon.py \
  --input-path modeling/data/processed/transition_dataset.csv \
  --include-proba
```

Salida por defecto:

- `modeling/outputs/predictions.csv`

## 5) Siguientes mejoras recomendadas

1. Separar entrenamiento y prueba por cromosoma o por gen para validar generalización real.
2. Probar `presets=high_quality` o `best_quality` con mayor `--time-limit`.
3. Versionar datasets y métricas por corrida (`outputs/run_YYYYMMDD_HHMM/`).
4. Añadir experiment tracking (MLflow/W&B) si quieres comparar muchos entrenamientos.
