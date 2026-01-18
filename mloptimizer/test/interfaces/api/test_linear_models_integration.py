import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet

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


# LogisticRegression tests
def test_logistic_regression(breast_cancer_data):
    """Test LogisticRegression optimization."""
    X, y = breast_cancer_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=LogisticRegression
    )

    opt = GeneticSearch(
        estimator_class=LogisticRegression,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, LogisticRegression)
    score = opt.score(X, y)
    assert 0.0 <= score <= 1.0


# Ridge tests
def test_ridge(diabetes_data):
    """Test Ridge regression optimization."""
    X, y = diabetes_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=Ridge
    )

    opt = GeneticSearch(
        estimator_class=Ridge,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, Ridge)


# Lasso tests
def test_lasso(diabetes_data):
    """Test Lasso regression optimization."""
    X, y = diabetes_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=Lasso
    )

    opt = GeneticSearch(
        estimator_class=Lasso,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, Lasso)


# ElasticNet tests
def test_elastic_net(diabetes_data):
    """Test ElasticNet regression optimization."""
    X, y = diabetes_data
    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=ElasticNet
    )

    opt = GeneticSearch(
        estimator_class=ElasticNet,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    assert opt.best_estimator_ is not None
    assert isinstance(opt.best_estimator_, ElasticNet)
