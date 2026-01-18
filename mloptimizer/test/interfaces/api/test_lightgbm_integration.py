import pytest
from sklearn.datasets import load_breast_cancer

from mloptimizer.interfaces.api import GeneticSearch, HyperparameterSpaceBuilder

# Import lightgbm - skip test if not available
lightgbm = pytest.importorskip("lightgbm")
from lightgbm import LGBMClassifier


@pytest.fixture
def breast_cancer_data():
    """Fixture to load the Breast Cancer dataset."""
    data = load_breast_cancer()
    return data.data, data.target


@pytest.fixture
def hyperparam_space_lgbm():
    """Fixture to load default hyperparameter space for LGBMClassifier."""
    return HyperparameterSpaceBuilder.get_default_space(estimator_class=LGBMClassifier)


@pytest.fixture
def genetic_optimizer_lgbm(hyperparam_space_lgbm):
    """Fixture to create a GeneticSearch instance for LGBMClassifier."""
    return GeneticSearch(
        estimator_class=LGBMClassifier,
        hyperparam_space=hyperparam_space_lgbm,
        generations=3,
        population_size=5,
        eval_function=None,
        seed=42,
        use_parallel=False
    )


def test_fit_lgbm(genetic_optimizer_lgbm, breast_cancer_data):
    """Test fitting the GeneticSearch with LGBMClassifier using the Breast Cancer dataset."""
    X, y = breast_cancer_data

    genetic_optimizer_lgbm.fit(X, y)

    # Validate that the best estimator and best parameters are set
    assert genetic_optimizer_lgbm.best_estimator_ is not None
    assert genetic_optimizer_lgbm.best_params_ is not None
    assert genetic_optimizer_lgbm.cv_results_ is not None

    # Verify that the best estimator is a LGBMClassifier
    assert isinstance(genetic_optimizer_lgbm.best_estimator_, LGBMClassifier)


def test_predict_lgbm(genetic_optimizer_lgbm, breast_cancer_data):
    """Test predicting with the fitted LGBMClassifier."""
    X, y = breast_cancer_data

    genetic_optimizer_lgbm.fit(X, y)
    predictions = genetic_optimizer_lgbm.predict(X)

    # Validate predictions
    assert predictions is not None
    assert len(predictions) == len(y)


def test_score_lgbm(genetic_optimizer_lgbm, breast_cancer_data):
    """Test scoring with the fitted LGBMClassifier."""
    X, y = breast_cancer_data

    genetic_optimizer_lgbm.fit(X, y)
    score = genetic_optimizer_lgbm.score(X, y)

    # Validate score
    assert score >= 0.0 and score <= 1.0  # Accuracy should be between 0 and 1


def test_lgbm_hyperparameters(hyperparam_space_lgbm):
    """Test that the LightGBM hyperparameter space contains expected parameters."""
    evolvable_params = hyperparam_space_lgbm.evolvable_hyperparams

    # Check that key LightGBM parameters are present
    assert "learning_rate" in evolvable_params
    assert "max_depth" in evolvable_params
    assert "n_estimators" in evolvable_params
    assert "num_leaves" in evolvable_params
    assert "subsample" in evolvable_params
    assert "colsample_bytree" in evolvable_params
    assert "reg_alpha" in evolvable_params
    assert "reg_lambda" in evolvable_params



def test_load_default_lgbm_space():
    """Test loading default hyperparameter space for LGBMClassifier."""
    space = HyperparameterSpaceBuilder.get_default_space(estimator_class=LGBMClassifier)

    assert space is not None
    assert len(space.evolvable_hyperparams) == 8  # Should have 8 evolvable parameters
