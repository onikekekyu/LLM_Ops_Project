"""Main constants of the project."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
env_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

# GCP Configuration
PROJECT_ID: str | None = os.getenv("GCP_PROJECT_ID")
REGION: str = os.getenv("GCP_REGION", "europe-west2")
BUCKET_NAME: str | None = os.getenv("GCP_BUCKET_NAME")
ENDPOINT_ID: str | None = os.getenv("GCP_ENDPOINT_ID")
PROJECT_NUMBER: str | None = os.getenv("GCP_PROJECT_NUMBER")

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: str | None = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

# Allow overriding the raw dataset URI directly via env var, otherwise build from bucket
RAW_DATASET_URI: str = os.getenv("RAW_DATASET_URI") or (
	f"gs://{BUCKET_NAME}/french_politics_sentences.csv" if BUCKET_NAME else ""
)

# Pipeline root path (used when running pipelines). Prefer an explicit env var.
PIPELINE_ROOT_PATH: str = os.getenv("PIPELINE_ROOT_PATH") or (
	f"{BUCKET_NAME}/vertexai-pipeline-root/" if BUCKET_NAME else ""
)
