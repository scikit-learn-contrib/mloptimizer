import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from mloptimizer.domain.hyperspace import HyperparameterSpace
from mloptimizer.interfaces.api import GeneticSearch, HyperparameterSpaceBuilder


@pytest.fixture
def iris_data():
    """Fixture to load the Iris dataset."""
    data = load_iris()
    return data.data, data.target


@pytest.fixture
def breast_cancer_data():
    """Fixture to load the Breast Cancer dataset."""
    data = load_breast_cancer()
    return data.data, data.target


@pytest.fixture
def hyperparam_space_rf():
    """Fixture to create a hyperparameter space for RandomForestClassifier."""
    builder = HyperparameterSpaceBuilder()
    hyperspace_rf = (builder.add_int_param("n_estimators", 10, 200)
                     .add_int_param("max_depth", 3, 20)
                     .add_int_param("min_samples_split", 2, 20)
                     .build())
    return hyperspace_rf


@pytest.fixture
def hyperparam_space_svc():
    """Fixture to create a hyperparameter space for SVC."""
    builder = HyperparameterSpaceBuilder()
    hyperspace_svc = (builder.add_float_param("C", 1, 1000)
                      .add_categorical_param("gamma", ["scale", "auto", 0.001, 0.01, 0.1, 1, 10])
                      .add_categorical_param("kernel", ["linear", "rbf", "poly"])
                      .build())
    return hyperspace_svc


@pytest.fixture
def hyperparam_space_dt():
    """Fixture to create a hyperparameter space for DecisionTreeClassifier."""
    builder = HyperparameterSpaceBuilder()
    hyperspace_dt = (builder.add_int_param("min_samples_split", 2, 20)
                     .add_int_param("max_depth", 2, 20)
                     .build())

    return hyperspace_dt


@pytest.fixture
def genetic_optimizer_rf(hyperparam_space_rf, iris_data):
    """Fixture to create a GeneticOptimizerAPI instance for RandomForestClassifier."""
    X, y = iris_data
    return GeneticSearch(
        estimator_class=RandomForestClassifier,
        hyperparam_space=hyperparam_space_rf,
        **{"generations": 5, "population_size": 5},
        eval_function=None,
        seed=42,
        use_parallel=False
    ).fit(X, y)


@pytest.fixture
def genetic_optimizer_svc(hyperparam_space_svc):
    """Fixture to create a GeneticOptimizerAPI instance for SVC."""
    return GeneticSearch(
        estimator_class=SVC,
        hyperparam_space=hyperparam_space_svc,
        **{"generations": 5, "population_size": 5},
        eval_function=None,
        seed=42,
        use_parallel=False
    )


@pytest.fixture
def genetic_optimizer_dt(hyperparam_space_dt):
    """Fixture to create a GeneticOptimizerAPI instance for DecisionTreeClassifier."""
    return GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=hyperparam_space_dt,
        **{"generations": 5, "population_size": 5},
        eval_function=None,
        seed=42,
        use_parallel=False
    )


def test_fit_random_forest(genetic_optimizer_rf, iris_data):
    """Test fitting the GeneticOptimizerAPI with RandomForestClassifier using the Iris dataset."""
    X, y = iris_data
    genetic_optimizer_rf.fit(X, y)

    # Validate that the best estimator and best parameters are set
    assert genetic_optimizer_rf.best_estimator_ is not None
    assert genetic_optimizer_rf.best_params_ is not None
    assert genetic_optimizer_rf.cv_results_ is not None


def test_fit_svc(genetic_optimizer_svc, breast_cancer_data):
    """Test fitting the GeneticOptimizerAPI with SVC using the Breast Cancer dataset."""
    X, y = breast_cancer_data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    genetic_optimizer_svc.fit(X, y)

    # Validate that the best estimator and best parameters are set
    assert genetic_optimizer_svc.best_estimator_ is not None
    assert genetic_optimizer_svc.best_params_ is not None
    assert genetic_optimizer_svc.cv_results_ is not None


def test_predict(genetic_optimizer_rf, iris_data):
    """Test predicting with the fitted model from GeneticOptimizerAPI with RandomForestClassifier."""
    X, y = iris_data
    genetic_optimizer_rf.fit(X, y)

    predictions = genetic_optimizer_rf.predict(X)

    # Validate predictions
    assert predictions is not None
    assert len(predictions) == len(y)


def test_score(genetic_optimizer_rf, iris_data):
    """Test scoring with the fitted model from GeneticOptimizerAPI with RandomForestClassifier."""
    X, y = iris_data
    genetic_optimizer_rf.fit(X, y)

    score = genetic_optimizer_rf.score(X, y)

    # Validate score
    assert score >= 0.0 and score <= 1.0  # Accuracy should be between 0 and 1


def test_set_hyperparameter_space(genetic_optimizer_rf, hyperparam_space_dt, iris_data):
    """Test setting a new hyperparameter space dynamically in GeneticOptimizerAPI."""
    X, y = iris_data

    # Set a new hyperparameter space (DecisionTreeClassifier space)
    genetic_optimizer_rf.set_hyperparameter_space(hyperparam_space_dt)

    # Fit with the new hyperparameter space
    genetic_optimizer_rf.fit(X, y)

    # Validate that the model was fitted with the new hyperparameter space
    assert genetic_optimizer_rf.best_estimator_ is not None
    assert genetic_optimizer_rf.best_params_ is not None


def test_save_hyperparameter_space(genetic_optimizer_rf, tmp_path):
    """Test saving the hyperparameter space."""
    hyperparam_space = genetic_optimizer_rf.hyperparam_space
    saved_hyperspace_path = tmp_path / f"{RandomForestClassifier.__name__.lower()}_hyperparameter_space.json"
    # Save the hyperparameter space to a temporary path
    genetic_optimizer_rf.save_hyperparameter_space(saved_hyperspace_path,
                                                   overwrite=True)

    # Validate that the hyperparameter space was saved
    assert saved_hyperspace_path.exists()


def test_load_default_hyperparameter_space(genetic_optimizer_rf):
    """Test loading a default hyperparameter space."""
    hyperparam_space = genetic_optimizer_rf.load_default_hyperparameter_space(RandomForestClassifier)

    # Validate the loaded hyperparameter space
    assert hyperparam_space is not None
    assert isinstance(hyperparam_space, HyperparameterSpace)


def test_load_hyperparameter_space(genetic_optimizer_rf, tmp_path):
    """Test loading the hyperparameter space."""
    # First we save the hyperparameter space
    hyperparam_space = genetic_optimizer_rf.hyperparam_space
    saved_hyperspace_path = tmp_path / f"{RandomForestClassifier.__name__.lower()}_hyperparameter_space.json"
    # Save the hyperparameter space to a temporary path
    genetic_optimizer_rf.save_hyperparameter_space(saved_hyperspace_path,
                                                   overwrite=True)

    # Validate that the hyperparameter space was saved
    assert saved_hyperspace_path.exists()

    # Load the saved hyperparameter space
    loaded_hyperparam_space = genetic_optimizer_rf.load_hyperparameter_space(saved_hyperspace_path)

    # Validate the loaded hyperparameter space
    assert loaded_hyperparam_space is not None
    assert isinstance(loaded_hyperparam_space, HyperparameterSpace)
    assert loaded_hyperparam_space == hyperparam_space


def test_get_genetic_params(genetic_optimizer_rf):
    """Test getting the genetic algorithm parameters."""
    genetic_params = genetic_optimizer_rf.get_genetic_params()

    # Validate that the genetic parameters are retrieved correctly
    assert genetic_params is not None
    assert isinstance(genetic_params, dict)
    assert genetic_params["generations"] == 5
    assert genetic_params["population_size"] == 5


def test_set_genetic_params(genetic_optimizer_rf):
    """Test setting the genetic algorithm parameters."""
    # Update genetic parameters
    new_genetic_params = {"generations": 10, "population_size": 8}
    genetic_optimizer_rf.set_params(**new_genetic_params)

    # Retrieve the updated genetic parameters
    updated_genetic_params = genetic_optimizer_rf.get_genetic_params()

    # Validate that the genetic parameters were updated correctly
    assert updated_genetic_params["generations"] == 10
    assert updated_genetic_params["population_size"] == 8


def test_verbose_parameter(iris_data, hyperparam_space_rf):
    """Test that verbose parameter is correctly set and returned in get_params."""
    X, y = iris_data

    # Test verbose=0 (default)
    opt = GeneticSearch(
        estimator_class=RandomForestClassifier,
        hyperparam_space=hyperparam_space_rf,
        generations=2,
        population_size=5,
        seed=42,
        verbose=0
    )
    assert opt.verbose == 0
    assert opt.get_params()["verbose"] == 0

    # Test verbose=1
    opt = GeneticSearch(
        estimator_class=RandomForestClassifier,
        hyperparam_space=hyperparam_space_rf,
        generations=2,
        population_size=5,
        seed=42,
        verbose=1
    )
    assert opt.verbose == 1
    assert opt.get_params()["verbose"] == 1

    # Test verbose=2
    opt = GeneticSearch(
        estimator_class=RandomForestClassifier,
        hyperparam_space=hyperparam_space_rf,
        generations=2,
        population_size=5,
        seed=42,
        verbose=2
    )
    assert opt.verbose == 2
    assert opt.get_params()["verbose"] == 2
