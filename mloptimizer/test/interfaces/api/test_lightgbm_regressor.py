import pytest
from sklearn.datasets import load_diabetes

from mloptimizer.interfaces.api import GeneticSearch, HyperparameterSpaceBuilder

# Import lightgbm - skip test if not available
lightgbm = pytest.importorskip("lightgbm")
from lightgbm import LGBMRegressor


@pytest.fixture
def diabetes_data():
    """Fixture to load the Diabetes dataset."""
    data = load_diabetes()
    return data.data, data.target


@pytest.fixture
def hyperparam_space_lgbmreg():
    """Fixture to load default hyperparameter space for LGBMRegressor."""
    return HyperparameterSpaceBuilder.get_default_space(estimator_class=LGBMRegressor)


def test_fit_lgbmregressor(hyperparam_space_lgbmreg, diabetes_data):
    """Test fitting the GeneticSearch with LGBMRegressor."""
    X, y = diabetes_data

    opt = GeneticSearch(
        estimator_class=LGBMRegressor,
        hyperparam_space=hyperparam_space_lgbmreg,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)

    assert opt.best_estimator_ is not None
    assert opt.best_params_ is not None
    assert opt.cv_results_ is not None
    assert isinstance(opt.best_estimator_, LGBMRegressor)


def test_predict_lgbmregressor(hyperparam_space_lgbmreg, diabetes_data):
    """Test predicting with the fitted LGBMRegressor."""
    X, y = diabetes_data

    opt = GeneticSearch(
        estimator_class=LGBMRegressor,
        hyperparam_space=hyperparam_space_lgbmreg,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)
    predictions = opt.predict(X)

    assert predictions is not None
    assert len(predictions) == len(y)
