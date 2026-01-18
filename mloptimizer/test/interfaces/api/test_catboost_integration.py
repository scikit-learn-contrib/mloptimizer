import pytest
from sklearn.datasets import load_breast_cancer

from mloptimizer.interfaces.api import GeneticSearch, HyperparameterSpaceBuilder

# Import catboost - skip test if not available
catboost = pytest.importorskip("catboost")
from catboost import CatBoostClassifier


@pytest.fixture
def breast_cancer_data():
    """Fixture to load the Breast Cancer dataset."""
    data = load_breast_cancer()
    return data.data, data.target


@pytest.fixture
def hyperparam_space_catboost():
    """Fixture to load default hyperparameter space for CatBoostClassifier."""
    return HyperparameterSpaceBuilder.get_default_space(estimator_class=CatBoostClassifier)


@pytest.fixture
def genetic_optimizer_catboost(hyperparam_space_catboost):
    """Fixture to create a GeneticSearch instance for CatBoostClassifier."""
    return GeneticSearch(
        estimator_class=CatBoostClassifier,
        hyperparam_space=hyperparam_space_catboost,
        generations=3,
        population_size=5,
        eval_function=None,
        seed=42,
        use_parallel=False
    )


def test_fit_catboost(genetic_optimizer_catboost, breast_cancer_data):
    """Test fitting the GeneticSearch with CatBoostClassifier using the Breast Cancer dataset."""
    X, y = breast_cancer_data

    genetic_optimizer_catboost.fit(X, y)

    # Validate that the best estimator and best parameters are set
    assert genetic_optimizer_catboost.best_estimator_ is not None
    assert genetic_optimizer_catboost.best_params_ is not None
    assert genetic_optimizer_catboost.cv_results_ is not None

    # Verify that the best estimator is a CatBoostClassifier
    assert isinstance(genetic_optimizer_catboost.best_estimator_, CatBoostClassifier)


def test_predict_catboost(genetic_optimizer_catboost, breast_cancer_data):
    """Test predicting with the fitted CatBoostClassifier."""
    X, y = breast_cancer_data

    genetic_optimizer_catboost.fit(X, y)
    predictions = genetic_optimizer_catboost.predict(X)

    # Validate predictions
    assert predictions is not None
    assert len(predictions) == len(y)


def test_score_catboost(genetic_optimizer_catboost, breast_cancer_data):
    """Test scoring with the fitted CatBoostClassifier."""
    X, y = breast_cancer_data

    genetic_optimizer_catboost.fit(X, y)
    score = genetic_optimizer_catboost.score(X, y)

    # Validate score
    assert score >= 0.0 and score <= 1.0  # Accuracy should be between 0 and 1


def test_catboost_hyperparameters(hyperparam_space_catboost):
    """Test that the CatBoost hyperparameter space contains expected parameters."""
    evolvable_params = hyperparam_space_catboost.evolvable_hyperparams

    # Check that key CatBoost parameters are present
    assert "learning_rate" in evolvable_params
    assert "depth" in evolvable_params
    assert "n_estimators" in evolvable_params
    assert "subsample" in evolvable_params
    assert "l2_leaf_reg" in evolvable_params
    assert "colsample_bylevel" in evolvable_params
    assert "random_strength" in evolvable_params

    # Check fixed parameters
    fixed_params = hyperparam_space_catboost.fixed_hyperparams
    assert "auto_class_weights" in fixed_params
    assert "bootstrap_type" in fixed_params
    assert "allow_writing_files" in fixed_params
    assert "verbose" in fixed_params
    assert "thread_count" in fixed_params
    assert fixed_params["allow_writing_files"] is False
    assert fixed_params["verbose"] is False
    assert fixed_params["thread_count"] == 1


def test_load_default_catboost_space():
    """Test loading default hyperparameter space for CatBoostClassifier."""
    space = HyperparameterSpaceBuilder.get_default_space(estimator_class=CatBoostClassifier)

    assert space is not None
    assert len(space.evolvable_hyperparams) == 7  # Should have 7 evolvable parameters
    assert len(space.fixed_hyperparams) == 5  # Should have 5 fixed parameters (including thread_count)
