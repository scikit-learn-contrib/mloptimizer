import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.neural_network import MLPClassifier, MLPRegressor

from mloptimizer.interfaces.api import GeneticSearch, HyperparameterSpaceBuilder


@pytest.fixture
def breast_cancer_data():
    """Fixture to load the Breast Cancer dataset."""
    data = load_breast_cancer()
    return data.data, data.target


@pytest.fixture
def diabetes_data():
    """Fixture to load the Diabetes dataset for regression."""
    data = load_diabetes()
    return data.data, data.target


# MLPClassifier tests
def test_mlp_classifier(breast_cancer_data):
    """Test MLPClassifier optimization."""
    X, y = breast_cancer_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=MLPClassifier
    )

    opt = GeneticSearch(
        estimator_class=MLPClassifier,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, MLPClassifier)
    score = opt.score(X, y)
    assert 0.0 <= score <= 1.0


# MLPRegressor tests
def test_mlp_regressor(diabetes_data):
    """Test MLPRegressor optimization."""
    X, y = diabetes_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=MLPRegressor
    )

    opt = GeneticSearch(
        estimator_class=MLPRegressor,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, MLPRegressor)
    predictions = opt.predict(X)
    assert predictions is not None
    assert len(predictions) == len(y)
