"""Script to register a model with a custom handler."""

from pathlib import Path

import typer
from google.cloud import aiplatform, storage

from src.constants import PROJECT_ID, REGION

# Handler path relative to the current script
HANDLER_PATH = Path(__file__).parents[1] / "src" / "handler.py"


def register_model_with_custom_handler(
    model_uri: str,
    display_name: str,
    parent_model: str | None = None,
    serving_container_image_uri: str = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cu121.2-3.transformers.4-46.ubuntu2204.py311",
    handler_path: Path = HANDLER_PATH,
):
    """Registers a model with a custom handler in Vertex AI."""
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Extract bucket name and blob path from GCS URI
    # Convert gs://bucket-name/path/to/model to (bucket-name, path/to/model)
    bucket_name = model_uri.replace("gs://", "").split("/")[0]
    blob_path = "/".join(model_uri.replace("gs://", "").split("/")[1:]) + "/handler.py"
    
    # Upload the custom handler to GCS
    (
        storage.Client()
        .bucket(bucket_name)
        .blob(blob_path)
        .upload_from_filename(str(handler_path))
    )

    # Register the model with the custom handler
    aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_ports=[8080],
        parent_model=parent_model,
    )


if __name__ == "__main__":
    typer.run(register_model_with_custom_handler)
