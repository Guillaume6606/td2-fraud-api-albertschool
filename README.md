# TD 2 — Service de scoring de fraude

Un modèle XGBoost **déjà entraîné** score des transactions de carte bancaire. Les données viennent de
FraudFinder, un jeu public de Google : 111 millions de transactions simulées, 2 % de fraude. Le
service est écrit avec FastAPI et livré avec son `Dockerfile`.

## Ce que fait le service

`POST /predict` reçoit des transactions et renvoie un score de fraude entre 0 et 1 pour chacune.

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "transactions": [{
    "tx_id": "8d2715f36c3b6983b28cb4931b4f3ad1d7af706d",
    "tx_ts": "2026-10-01T09:02:07",
    "customer_id": "3540440915461489",
    "terminal_id": "00064542",
    "tx_amount": 22.48
  }]
}'
```

Le score dépend de 20 variables, calculées à partir de la transaction et de compteurs journaliers :

| Famille | Variables | Source |
| --- | --- | --- |
| Transaction | montant, heure, jour de la semaine | la demande |
| Client | nombre de transactions et montant moyen sur 1, 7 et 14 jours, rapport du montant à cette moyenne | `data/customer_daily.parquet` |
| Terminal | nombre de transactions sur 1, 7 et 14 jours, taux de fraude connu avec 7 jours de retard | `data/terminal_daily.parquet` |

Les compteurs s'arrêtent au 30 septembre 2026. Les transactions à scorer commencent le 1er octobre
2026 (`data/replay.parquet`).

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `app.py` | service FastAPI |
| `features.py` | compteurs journaliers et calcul des 20 variables |
| `scoring.py` | chargement du modèle, calcul des scores |
| `settings.py` | variables `DATA_DIR`, `MODELS_DIR`, `XGB_NTHREAD` |
| `models/` | modèle, liste des variables, scores de référence |
| `tests/` | tests du service |

Le modèle est dans l'image. Les fichiers de `data/` n'y sont pas : le service les lit dans le dossier
`DATA_DIR`.

## Installation

```bash
uv sync
uv run uvicorn app:app --port 8000
```

Placez dans `data/` les trois fichiers parquet indiqués par l'enseignant.

## Image

```bash
docker build --platform linux/amd64 -t fraud-api .
docker run --rm -p 8000:8000 -v "$PWD/data:/app/data:ro" fraud-api
```
