import pytest
from sklearn.datasets import load_diabetes

from mloptimizer.interfaces.api import GeneticSearch, HyperparameterSpaceBuilder

# Import catboost - skip test if not available
catboost = pytest.importorskip("catboost")
from catboost import CatBoostRegressor


@pytest.fixture
def diabetes_data():
    """Fixture to load the Diabetes dataset."""
    data = load_diabetes()
    return data.data, data.target


@pytest.fixture
def hyperparam_space_catboostreg():
    """Fixture to load default hyperparameter space for CatBoostRegressor."""
    return HyperparameterSpaceBuilder.get_default_space(estimator_class=CatBoostRegressor)


def test_fit_catboostregressor(hyperparam_space_catboostreg, diabetes_data):
    """Test fitting the GeneticSearch with CatBoostRegressor."""
    X, y = diabetes_data

    opt = GeneticSearch(
        estimator_class=CatBoostRegressor,
        hyperparam_space=hyperparam_space_catboostreg,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)

    assert opt.best_estimator_ is not None
    assert opt.best_params_ is not None
    assert opt.cv_results_ is not None
    assert isinstance(opt.best_estimator_, CatBoostRegressor)


def test_predict_catboostregressor(hyperparam_space_catboostreg, diabetes_data):
    """Test predicting with the fitted CatBoostRegressor."""
    X, y = diabetes_data

    opt = GeneticSearch(
        estimator_class=CatBoostRegressor,
        hyperparam_space=hyperparam_space_catboostreg,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    predictions = opt.predict(X)

    assert predictions is not None
    assert len(predictions) == len(y)
