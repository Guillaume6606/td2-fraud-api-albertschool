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
| `app.py` | service FastAPI, **à reprendre à l'étape 0** |
| `features.py` | compteurs journaliers et calcul des 20 variables |
| `scoring.py` | chargement du modèle, calcul des scores |
| `settings.py` | variables `DATA_DIR`, `MODELS_DIR`, `XGB_NTHREAD` |
| `models/` | modèle, liste des variables, scores de référence |
| `tests/` | ce que le service doit vérifier à la fin de l'étape 0 |

Le modèle est dans l'image. Les fichiers de `data/` n'y sont pas : le service les lit dans le dossier
`DATA_DIR`.

## Installation

```bash
uv sync
uv run uvicorn app:app --port 8000
```

Placez dans `data/` les trois fichiers parquet indiqués par l'enseignant.

## Étape 0 — Mettre le service à niveau

Le service de départ répond juste, mais :

1. il relit le modèle et les compteurs à chaque demande ;
2. il ne vérifie pas le contenu des demandes ;
3. il n'a pas de point de contrôle de santé.

À faire dans `app.py` :

1. charger le modèle et les compteurs une seule fois, au démarrage ;
2. décrire la demande et la réponse avec des schémas Pydantic : une transaction invalide doit
   recevoir un code 422, et les noms en majuscules de FraudFinder (`TX_ID`, `TX_TS`, `CUSTOMER_ID`,
   `TERMINAL_ID`, `TX_AMOUNT`) doivent être acceptés ;
3. ajouter `GET /health`, qui renvoie `status` (`"ok"`), `features` (nombre de variables),
   `customers` et `terminals` (nombre d'identifiants chargés) et `last_day` (dernier jour connu) ;
4. ajouter à la réponse de `/predict` le champ `timings_ms` avec les durées `features`, `predict`
   et `handler` en millisecondes.

L'étape est finie quand `uv run pytest` passe. Mesurez la durée d'une demande avant et après.

## Image

```bash
docker build --platform linux/amd64 -t fraud-api .
docker run --rm -p 8000:8000 -v "$PWD/data:/app/data:ro" fraud-api
```
