import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor, GradientBoostingRegressor
)

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


# HistGradientBoostingClassifier tests
def test_hist_gradient_boosting_classifier(breast_cancer_data):
    """Test HistGradientBoostingClassifier optimization."""
    X, y = breast_cancer_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=HistGradientBoostingClassifier
    )

    opt = GeneticSearch(
        estimator_class=HistGradientBoostingClassifier,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, HistGradientBoostingClassifier)
    score = opt.score(X, y)
    assert 0.0 <= score <= 1.0


# HistGradientBoostingRegressor tests
def test_hist_gradient_boosting_regressor(diabetes_data):
    """Test HistGradientBoostingRegressor optimization."""
    X, y = diabetes_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=HistGradientBoostingRegressor
    )

    opt = GeneticSearch(
        estimator_class=HistGradientBoostingRegressor,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, HistGradientBoostingRegressor)
    predictions = opt.predict(X)
    assert predictions is not None
    assert len(predictions) == len(y)


# AdaBoostClassifier tests
def test_adaboost_classifier(breast_cancer_data):
    """Test AdaBoostClassifier optimization."""
    X, y = breast_cancer_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=AdaBoostClassifier
    )

    opt = GeneticSearch(
        estimator_class=AdaBoostClassifier,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, AdaBoostClassifier)


# AdaBoostRegressor tests
def test_adaboost_regressor(diabetes_data):
    """Test AdaBoostRegressor optimization."""
    X, y = diabetes_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=AdaBoostRegressor
    )

    opt = GeneticSearch(
        estimator_class=AdaBoostRegressor,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, AdaBoostRegressor)


# GradientBoostingRegressor tests
def test_gradient_boosting_regressor(diabetes_data):
    """Test GradientBoostingRegressor optimization."""
    X, y = diabetes_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=GradientBoostingRegressor
    )

    opt = GeneticSearch(
        estimator_class=GradientBoostingRegressor,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, GradientBoostingRegressor)
