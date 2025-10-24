# Projet LLM_Ops_Project

## Pré-requis du projet

### Gestion de l'environnement Python (uv)

- Il est recommandé d'utiliser `uv` pour gérer la version de Python, les environnements virtuels et la synchronisation des dépendances.
- Documentation : https://github.com/astral-sh/uv#installation

## Installation

### Environnement virtuel et dépendances

Exécutez l'une des commandes suivantes selon votre méthode :

```bash
# Avec uv (recommandé si configuré pour le projet)
uv sync

# Ou manuellement avec venv + pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # si présent
```

## Configuration GCP (prérequis)

1. Installer le SDK Google Cloud : https://cloud.google.com/sdk/docs/install
2. Créer ou identifier un projet GCP pour le projet.

### Authentification et configuration du projet

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Activer les APIs requises

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage-component.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
```

## Projet LLM_Ops_Project — Matériels et pipeline d'entraînement

Ce dépôt contient les composants et scripts d'un pipeline d'entraînement et d'inférence pour un projet pédagogique LLM/Ops.
Le but principal est de fournir une base reproductible pour :
- préparer/transformer des jeux de données,
- entraîner et évaluer un modèle (fine-tuning),
- déployer / tester l'inférence,
- valider la configuration GCP et automatiser les étapes via des scripts.

Ce README décrit comment préparer l'environnement, configurer les variables d'environnement, lancer les scripts principaux et comprendre l'organisation du dépôt.

## Prérequis

- macOS, Linux ou Windows WSL
- Python 3.10+ (ou la version définie dans le projet). Recommandation : utiliser `uv` pour gérer l'environnement (voir ci-dessous).
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (pour les étapes GCP)
- Docker et docker-compose (si vous utilisez `docker-compose.yml` fourni)

Optionnel : accès aux services Langfuse/MinIO/Postgres/ClickHouse si vous souhaitez activer la partie monitoring/observabilité locale.

## Installation rapide

1. Installer `uv` (optionnel, recommandé si le dépôt contient un fichier de configuration `uv`)

   Voir : https://github.com/astral-sh/uv#installation

2. Synchroniser l'environnement Python et installer les dépendances :

```bash
uv sync
# ou, si vous n'utilisez pas uv, créer un venv et installer avec pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # si le projet fournit requirements
```

3. Construire et lancer les services locaux (optionnel) :

```bash
docker-compose up -d
```

## Variables d'environnement

Copiez `.env.example` en `.env` et remplissez les valeurs nécessaires :

```bash
cp .env.example .env
# Éditez .env et renseignez vos identifiants/valeurs
```

Principales variables présentes dans `.env.example` :

- PYTHONPATH : ajoute la racine du projet au chemin Python.
- GCP_PROJECT_ID : identifiant du projet GCP.
- GCP_REGION : région GCP (ex. europe-west2).
- GCP_BUCKET_NAME : nom du bucket GCS utilisé pour les données.
- RAW_DATASET_URI : URI du fichier de données dans GCS (ex. gs://bucket/name.csv).
- GCP_PROJECT_NUMBER : numéro du projet GCP (optionnel selon scripts).
- GCP_ENDPOINT_ID : identifiant d'un endpoint AI Platform (si utilisé).

- LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_HOST : configuration Langfuse (observabilité). Remplacez par vos clés ou laissez vides en local.

- NEXTAUTH_SECRET, SALT, ENCRYPTION_KEY : secrets pour authentification/chiffrement (ex. services locaux). Ne pas committer de vraies clés publiques.

- POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, DATABASE_URL : configuration base de données locale (ex. pour Langfuse).

- CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, REDIS_AUTH : identifiants pour stacks locaux.

Remplissez `.env` avec des valeurs réelles avant d'exécuter les scripts qui en dépendent.

## Scripts principaux

Les scripts utiles se trouvent dans le dossier `scripts/` :

- `validate_gcp_setup.py` : vérifie que les ressources et configurations GCP requises sont accessibles et actives.
- `pipeline_runner.py` : point d'entrée pour lancer le pipeline d'entraînement (orchestration locale).
- `register_model_with_custom_handler.py` : script d'exemple pour enregistrer un modèle et son handler personnalisé.

Exemples d'utilisation :

```bash
# 1) Valider la configuration GCP (après avoir renseigné GCP_PROJECT_ID dans .env)
gcloud auth login
gcloud auth application-default login
python scripts/validate_gcp_setup.py

# 2) Lancer le pipeline d'entraînement (orchestration locale)
python scripts/pipeline_runner.py --config model_training_pipeline.json

# 3) Enregistrer un modèle avec le handler personnalisé (exemple)
python -m scripts.register_model_with_custom_handler \
  "gs://llm-ops-bucket-kiki/vertexai-pipeline-root/54825872111/model-training-pipeline-20251022101231/fine-tuning-component_174482324845494272/model" \
  "kiki-french-politics-phi3"
# 4) Lancer l'interface Chainlit localement pour tester l'application de chat
# Assurez-vous d'être dans l'environnement virtuel et d'avoir installé les dépendances
python -m chainlit run src/app/main.py                
# Par défaut Chainlit démarre l'interface web sur http://localhost:8000

# 5) Voir le monitoring avec Langfuse (stack local via docker-compose)
# Démarrer les services locaux (Postgres, MinIO, Langfuse, worker, etc.)
docker-compose up -d
# Ouvrez ensuite l'interface Langfuse (par défaut) :
# http://localhost:3000
# Remarque : pour envoyer des évènements à Langfuse, renseignez LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY dans .env
```

## Structure du projet

Principaux dossiers/fichiers :

- `src/` : code source du projet
  - `app/` : application ou point d'entrée (ex. `main.py`)
  - `pipeline_components/` : composants du pipeline (prétraitement, fine-tuning, évaluation, inference)
  - `pipelines/` : définitions des pipelines (p. ex. `model_training_pipeline.py`)
  - `handler.py` : handler d'inférence personnalisé
  - `monitoring.py` : utilitaires de monitoring/observabilité
- `scripts/` : scripts utilitaires pour lancer/valider/enregistrer
- `docker-compose.yml` : configuration pour exécuter les services locaux (Postgres, MinIO, ClickHouse, etc.)
- `model_training_pipeline.json` : configuration de pipeline par défaut

## Bonnes pratiques et sécurité

- Ne commitez jamais de secrets réels (clés API, mots de passe). Le fichier `.env` doit rester local et ignoré par Git (ajouté par défaut au `.gitignore`).
- Utilisez `.env.example` pour documenter les variables nécessaires sans exposer de valeurs sensibles.

## Dépannage

- Problème d'authentification GCP : assurez-vous d'avoir exécuté `gcloud auth login` et `gcloud auth application-default login`.
- Erreurs liées aux dépendances : activez le venv puis réinstallez les dépendances.
- Services locaux indisponibles : vérifiez `docker-compose ps` et les logs (`docker-compose logs -f`).



