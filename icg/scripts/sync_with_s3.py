from __future__ import annotations
import os
import tempfile
from typing import Dict, Any, Optional
import yaml

import boto3
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from pathlib import Path

import sys
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def sync_model(
    data: Dict[str, Any],
    s3_bucket: Optional[str] =  "icg-bucket-mlops",
    s3_prefix: str = "Production-model",
    aws_region: Optional[str] = None,
) -> Dict[str, Any]:
    
    CONFIG_PATH = "/opt/airflow/icg/config.yaml"
    cfg = _load_yaml(CONFIG_PATH)

    sync_cfg = cfg.get("sync_cfg", {})

    s3_bucket = sync_cfg.get('s3_bucket',s3_bucket)
    s3_prefix = sync_cfg.get('s3_prefix',s3_prefix)
    aws_region = sync_cfg.get('aws_region',aws_region)

    model_name = data["model_name"]

    # --- Get Production model version from MLflow registry ---
    client = MlflowClient()
    latest = client.get_latest_versions(model_name, stages=["Production"])
    if not latest:
        raise RuntimeError(f"No Production model found for '{model_name}' in MLflow registry.")

    prod_version = latest[0].version
    run_id = latest[0].run_id

    # --- Resolve S3 bucket ---
    bucket = (
        s3_bucket
        or data.get("s3_bucket")
    )
    if not bucket:
        raise RuntimeError(
            "S3 bucket not provided. Pass s3_bucket=..., or set env S3_BUCKET / ML_S3_BUCKET."
        )

    # --- Download model from MLflow (local temp dir) ---
    artifact_uri = f"models:/{model_name}/{prod_version}"
    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = download_artifacts(artifact_uri=artifact_uri, dst_path=tmpdir)

        # Find the .pth file inside the downloaded model directory
        model_pth = None
        for root, _, files in os.walk(local_dir):
            for f in files:
                if f.endswith(".pth"):
                    model_pth = os.path.join(root, f)
                    break
            if model_pth:
                break

        if not model_pth:
            raise FileNotFoundError(
                f"No .pth file found in the downloaded model artifacts for {model_name} v{prod_version}"
            )
        
        vocab_json = "./data/Flickr8k/artifacts/vocab.json"
        
        # --- Upload to S3 ---
        s3 = boto3.client("s3", region_name=aws_region)
        model_key = f"{s3_prefix.rstrip('/')}/model.pth"
        version_key = f"{s3_prefix.rstrip('/')}/version.txt"
        vocab_key = f"{s3_prefix.rstrip('/')}/vocab.json"

        # Upload model weights
        s3.upload_file(model_pth, bucket, model_key)

        s3.upload_file(
            vocab_json, bucket, vocab_key, ExtraArgs={"ContentType": "application/json"}
        )

        # Upload version info
        s3.put_object(Bucket=bucket, Key=version_key, Body=str(prod_version).encode("utf-8"))


    return {
        "model_name": model_name,
        "production_version": prod_version,
        "run_id": run_id,
        "s3_bucket": bucket,
        "s3_prefix": s3_prefix,
        "uploaded": {
            "model_pth": model_key,
            "version_txt": version_key,
        },
    }
