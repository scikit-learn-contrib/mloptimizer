import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mloptimizer.interfaces.api import GeneticSearch, HyperparameterSpaceBuilder


@pytest.fixture
def iris_data():
    """Fixture to load the Iris dataset."""
    data = load_iris()
    return data.data, data.target


def test_n_trials_attribute(iris_data):
    """Test that n_trials_ attribute is set correctly after fitting."""
    X, y = iris_data

    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=DecisionTreeClassifier
    )

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=hyperparam_space,
        generations=3,
        population_size=5,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)

    # Check that n_trials_ attribute exists and is a positive integer
    assert hasattr(opt, 'n_trials_')
    assert isinstance(opt.n_trials_, int)
    assert opt.n_trials_ > 0

    # For 3 generations with population size 5, we expect fewer than 15 evaluations
    # due to elitism (best individuals are carried over with cached fitness)
    # We should have at least the initial population evaluated
    assert opt.n_trials_ >= 5
    # And less than total population count (due to fitness caching)
    assert opt.n_trials_ < len(opt.populations_)


def test_optimization_time_attribute(iris_data):
    """Test that optimization_time_ attribute is set correctly after fitting."""
    X, y = iris_data

    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=DecisionTreeClassifier
    )

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=hyperparam_space,
        generations=2,
        population_size=4,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)

    # Check that optimization_time_ attribute exists and is a positive float
    assert hasattr(opt, 'optimization_time_')
    assert isinstance(opt.optimization_time_, float)
    assert opt.optimization_time_ > 0

    # Sanity check: optimization should not take more than 60 seconds for this small test
    assert opt.optimization_time_ < 60.0


def test_performance_attributes_consistency(iris_data):
    """Test that n_trials_ is consistent with logbook and less than total population."""
    X, y = iris_data

    hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
        estimator_class=DecisionTreeClassifier
    )

    opt = GeneticSearch(
        estimator_class=DecisionTreeClassifier,
        hyperparam_space=hyperparam_space,
        generations=3,
        population_size=6,
        seed=42,
        use_parallel=False
    )

    opt.fit(X, y)

    # n_trials_ should equal the sum of nevals from logbook
    # (actual evaluations, excluding cached fitness from elitism)
    assert opt.n_trials_ == sum(record['nevals'] for record in opt.logbook_)

    # n_trials_ should be less than total population count due to elitism
    # (elites are carried over with cached fitness and not re-evaluated)
    assert opt.n_trials_ < len(opt.populations_)

    # Both attributes should be set
    assert opt.n_trials_ is not None
    assert opt.optimization_time_ is not None
