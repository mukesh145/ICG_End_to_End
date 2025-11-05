from __future__ import annotations
import os
from typing import Dict, Any, Optional

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


def compare_and_promote_model(data: Dict[str, Any]) -> Dict[str, Any]:
    
    if not isinstance(data, dict):
        raise ValueError("`data` must be a dict containing at least the key 'run_id'.")

    run_id = data.get("run_id")

    if not run_id:
        raise ValueError("`data['run_id']` is required.")

    model_name: Optional[str] = data.get("model_name") 
    if not model_name:
        raise ValueError(
            "Model name not provided. Pass `data['model_name']` or set env var MLFLOW_MODEL_NAME."
        )

    client = MlflowClient()

    # --- Get candidate run & metric ---
    cand_run = client.get_run(run_id)

    if "val_loss" not in cand_run.data.metrics:
        raise ValueError(
            f"Run {run_id} has no 'val_loss' metric. Ensure you logged it before calling this."
        )
    
    cand_val_loss = cand_run.data.metrics["val_loss"]


    mvs = client.search_model_versions(f"name = '{model_name}' and run_id = '{run_id}'")

    if not mvs:
        raise MlflowException(
            f"No registered Model Version found for name='{model_name}' with run_id='{run_id}'. "
            "Register the run as a model version first."
        )
    
    cand_mv = max(mvs, key=lambda mv: int(mv.version))  # pick the latest version attached to this run
    cand_version = int(cand_mv.version)
    print(f"Candidate Version : {cand_version}")

    # --- Get current Production model version (if any) and its metric ---
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    print(f"Prod Version : {prod_versions}")

    prod_mv = prod_versions[0] if prod_versions else None

    prod_version_before = int(prod_mv.version) if prod_mv else None
    prod_val_loss = None
    if prod_mv is not None:
        prod_run = client.get_run(prod_mv.run_id)
        if "val_loss" not in prod_run.data.metrics:
            raise ValueError(
                f"Current Production version (v{prod_mv.version}) has no 'val_loss' metric."
            )
        prod_val_loss = prod_run.data.metrics["val_loss"]

    # --- Decide & transition ---
    promoted = False
    if prod_mv is None:
        # No Production yet -> promote candidate
        client.transition_model_version_stage(
            name=model_name,
            version=str(cand_version),
            stage="Production",
            archive_existing_versions=True,  # nothing to archive, but harmless
        )
        promoted = True
        prod_version_after = cand_version
    else:
        if cand_val_loss < prod_val_loss:
            # Promote candidate and auto-archive any existing Production versions
            client.transition_model_version_stage(
                name=model_name,
                version=str(cand_version),
                stage="Production",
                archive_existing_versions=True,
            )
            promoted = True
            prod_version_after = cand_version
        else:
            # Keep current Production; optionally move candidate to Staging (optional)
            prod_version_after = prod_version_before

    # --- Optional: add helpful descriptions for traceability ---
    try:
        if promoted:
            client.update_model_version(
                name=model_name,
                version=str(cand_version),
                description=f"Promoted to Production via compare_and_promote_model(). "
                            f"val_loss={cand_val_loss:.6f}."
            )
            if prod_mv is not None:
                client.update_model_version(
                    name=model_name,
                    version=str(prod_mv.version),
                    description=f"Auto-archived by compare_and_promote_model(). "
                                f"val_loss={prod_val_loss:.6f}."
                )
        else:
            client.update_model_version(
                name=model_name,
                version=str(cand_version),
                description=f"Not promoted. Candidate val_loss={cand_val_loss:.6f} "
                            f">= Production val_loss={prod_val_loss:.6f}."
            )
    except Exception:
        # Don't fail the flow for description write errors
        pass
    
    info = {
        "model_name": model_name,
        "candidate": {
            "version": cand_version,
            "run_id": run_id,
            "val_loss": cand_val_loss,
        },
        "production_before": {
            "version": prod_version_before,
            "val_loss": prod_val_loss,
        },
        "promoted": promoted,
        "production_after": {
            "version": prod_version_after,
            "val_loss": (cand_val_loss if promoted else prod_val_loss),
        },
    }

    print(info)

    return info