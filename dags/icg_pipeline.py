# dags/ml_pipeline.py
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from pathlib import Path
import sys

default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

@dag(
    start_date=datetime(2025, 10, 10),
    schedule="@daily",
    catchup=False,
    tags=["ml"],
    default_args=default_args,
)
def ml_pipe_line():

    # ---------- Helpers to add project to sys.path ----------
    def _ensure_project_on_path():
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

    # ---------- Tasks ----------
    @task(task_id="fetch_data")
    def fetch_and_prepare_data():
        _ensure_project_on_path()
        from icg.scripts.prepare_dataset import prepare_entrypoint
        CONFIG_PATH = "/opt/airflow/icg/config.yaml"
        return prepare_entrypoint(CONFIG_PATH)

    @task(task_id="train_model")
    def train_model(data: dict) -> dict:
        _ensure_project_on_path()
        from icg.scripts.train_model import train_entrypoint
        CONFIG_PATH = "/opt/airflow/icg/config.yaml"
        return train_entrypoint(CONFIG_PATH)

    @task(task_id="compare_and_promote")
    def compare_and_promote(data: dict) -> dict:
        _ensure_project_on_path()
        from icg.scripts.compare_and_promote import compare_and_promote_model
        return compare_and_promote_model(data)

    @task.branch(task_id="trigger_sync")
    def trigger_sync(info: dict) -> str:
        # Must return the *task_id* of an immediate downstream task
        return "sync_with_s3" if info.get("promoted") else "end_pipeline"

    @task(task_id="sync_with_s3")
    def sync_with_s3(info: dict) -> dict:
        _ensure_project_on_path()
        from icg.scripts.sync_with_s3 import sync_model
        return sync_model(info)

    @task(task_id="end_pipeline")
    def end_pipeline(info: dict) -> dict:
        # No-op; useful for the "not promoted" path
        return {"status": "no_promotion", **info}

    @task(task_id="done", trigger_rule="none_failed_min_one_success")
    def done():
        return "pipeline_complete"

    # ---------- Graph wiring ----------
    data = fetch_and_prepare_data()
    trained = train_model(data)
    info = compare_and_promote(trained)

    branch = trigger_sync(info)
    
    sync_task = sync_with_s3(info)
    end_task = end_pipeline(info)

    # Branch to one of these:
    branch >> [sync_task, end_task]
    # Re-join safely:
    [sync_task, end_task] >> done()

ml_pipe_line()
