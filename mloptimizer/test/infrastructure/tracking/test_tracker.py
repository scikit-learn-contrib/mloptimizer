import logging
import pytest
from mloptimizer.infrastructure.tracking import Tracker
from sklearn.tree import DecisionTreeClassifier


# Set up a fixture for the Tracker instance
@pytest.fixture
def tracker_instance(tmp_path):
    name = "test_optimization"
    folder = tmp_path
    use_mlflow = False
    return Tracker(name, str(folder), use_mlflow=use_mlflow)


@pytest.fixture
def initialized_tracker_instance(tmp_path):
    name = "test_optimization"
    folder = tmp_path
    use_mlflow = False
    tracker_instance = Tracker(name, str(folder), use_mlflow=use_mlflow)
    tracker_instance.start_optimization("TestOptClass", 5)
    opt_run_folder_name = "checkpoint_test"
    tracker_instance.start_checkpoint(opt_run_folder_name, DecisionTreeClassifier)
    return tracker_instance


def test_tracker_init(tracker_instance, tmp_path):
    assert tracker_instance.name == "test_optimization"
    assert tracker_instance.folder == str(tmp_path)
    assert not tracker_instance.use_mlflow


def test_start_optimization(tracker_instance, caplog):
    # Enable logging capture for mloptimizer
    with caplog.at_level(logging.INFO, logger="mloptimizer"):
        tracker_instance.start_optimization("TestOptClass", 5)
    assert "Starting Genetic Algorithm Optimization" in caplog.text
    assert "Optimizer: TestOptClass" in caplog.text


def test_start_checkpoint_creates_directories(tracker_instance, tmp_path):
    opt_run_folder_name = "checkpoint_test"
    tracker_instance.start_checkpoint(opt_run_folder_name, DecisionTreeClassifier)
    expected_dirs = ["checkpoints", "results", "graphics", "progress"]
    for dir_name in expected_dirs:
        assert (tmp_path / opt_run_folder_name / dir_name).exists()


def test_log_clfs(tracker_instance, caplog):
    classifiers_list = [DecisionTreeClassifier(max_depth=30), DecisionTreeClassifier()]
    generation = 1
    fitness_list = [0.9, 0.8]
    tracker_instance.start_optimization("TestOptClass", 5)
    opt_run_folder_name = "checkpoint_test"
    tracker_instance.start_checkpoint(opt_run_folder_name, DecisionTreeClassifier)
    with caplog.at_level(logging.INFO, logger="mloptimizer"):
        tracker_instance.log_clfs(classifiers_list, generation, fitness_list)
    assert "Generation 1 - Classifier TOP 0" in caplog.text
    assert f"Classifier: {str(classifiers_list[0])}" in caplog.text
    assert "Fitness: 0.9" in caplog.text
