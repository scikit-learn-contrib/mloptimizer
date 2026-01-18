import subprocess
import socket
import sys
import time
import pytest
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking.client import MlflowClient
from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder

MLFLOW_PORT = 5001
MLFLOW_TRACKING_URI = f"http://127.0.0.1:{MLFLOW_PORT}"


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0

@pytest.fixture(scope="session")
def mlflow_server():
    """Start MLflow server for testing if not already running."""
    db_path = Path("mlflow_test.db").resolve()
    artifact_root = Path("mlruns_test").resolve()

    if not is_port_open(MLFLOW_PORT):
        proc = subprocess.Popen([
            sys.executable, "-m", "mlflow", "server",
            "--backend-store-uri", f"sqlite:///{db_path}",
            "--default-artifact-root", str(artifact_root),
            "--host", "127.0.0.1",
            "--port", str(MLFLOW_PORT)
        ])
        time.sleep(5)
        yield proc
        proc.terminate()
    else:
        yield None


def test_genetic_search_creates_mlflow_runs(mlflow_server):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Prepare dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Build hyperparameter space
    space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    # Run search with MLflow logging
    search = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=space,
        scoring="accuracy",
        **{"generations": 2, "population_size": 3},
        use_mlflow=True
    )
    search.fit(X_train, y_train)

    # Check MLflow logs
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiments = client.search_experiments()
    assert experiments, "No experiments found"

    # Find the latest experiment (assumes recent test run)
    latest_exp = sorted(experiments, key=lambda e: e.creation_time, reverse=True)[0]
    runs = client.search_runs([latest_exp.experiment_id])
    assert runs, "No runs found"

    parent_runs = [r for r in runs if r.data.tags.get("mlflow.parentRunId") is None]
    child_runs = [r for r in runs if r.data.tags.get("mlflow.parentRunId") is not None]

    assert parent_runs, "No parent MLflow run created"
    assert child_runs, "No nested MLflow runs created"

    # Confirm logging
    child = child_runs[0]
    assert child.data.params, "No parameters logged in child run"
    assert child.data.metrics, "No metrics logged in child run"

    # Confirm run names are formatted
    assert "gen_" in child.data.tags.get("mlflow.runName", ""), "Run name format incorrect"
