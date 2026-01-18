import pytest
from sklearn.datasets import load_diabetes

from mloptimizer.interfaces.api import GeneticSearch, HyperparameterSpaceBuilder

# Import xgboost - skip test if not available
xgboost = pytest.importorskip("xgboost")
from xgboost import XGBRegressor


@pytest.fixture
def diabetes_data():
    """Fixture to load the Diabetes dataset."""
    data = load_diabetes()
    return data.data, data.target


@pytest.fixture
def hyperparam_space_xgbreg():
    """Fixture to load default hyperparameter space for XGBRegressor."""
    return HyperparameterSpaceBuilder.get_default_space(estimator_class=XGBRegressor)


def test_fit_xgbregressor(hyperparam_space_xgbreg, diabetes_data):
    """Test fitting the GeneticSearch with XGBRegressor."""
    X, y = diabetes_data

    opt = GeneticSearch(
        estimator_class=XGBRegressor,
        hyperparam_space=hyperparam_space_xgbreg,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)

    assert opt.best_estimator_ is not None
    assert opt.best_params_ is not None
    assert opt.cv_results_ is not None
    assert isinstance(opt.best_estimator_, XGBRegressor)


def test_predict_xgbregressor(hyperparam_space_xgbreg, diabetes_data):
    """Test predicting with the fitted XGBRegressor."""
    X, y = diabetes_data

    opt = GeneticSearch(
        estimator_class=XGBRegressor,
        hyperparam_space=hyperparam_space_xgbreg,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    predictions = opt.predict(X)

    assert predictions is not None
    assert len(predictions) == len(y)
