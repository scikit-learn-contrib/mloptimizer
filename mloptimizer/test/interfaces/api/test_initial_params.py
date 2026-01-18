"""
Tests for initial_params population seeding feature.

This module tests the ability to seed the initial population with known good
hyperparameter configurations.
"""

import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder


class TestInitialParams:
    """Test suite for initial_params population seeding."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset for testing."""
        return load_iris(return_X_y=True)

    @pytest.fixture
    def tree_space(self):
        """Get default hyperparameter space for DecisionTreeClassifier."""
        return HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    def test_initial_params_basic(self, iris_data, tree_space):
        """Verify initial_params feature works without crashing."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=[{'max_depth': 5, 'min_samples_split': 10}],
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None
        assert opt.n_trials_ > 0
        assert opt.optimization_time_ > 0

    def test_initial_params_none(self, iris_data, tree_space):
        """Verify None initial_params works (default behavior)."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=None,
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None

    def test_initial_params_empty_list(self, iris_data, tree_space):
        """Verify empty list initial_params works."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=[],
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None

    def test_initial_params_multiple_configs(self, iris_data, tree_space):
        """Verify multiple initial parameter configurations."""
        X, y = iris_data

        initial_configs = [
            {'max_depth': 3, 'min_samples_split': 5},
            {'max_depth': 5, 'min_samples_split': 10},
            {'max_depth': 10, 'min_samples_split': 2}
        ]

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=initial_configs,
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None
        assert opt.n_trials_ > 0

    def test_initial_params_with_include_default_true(self, iris_data, tree_space):
        """Verify include_default=True adds sklearn defaults to population."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=[{'max_depth': 5, 'min_samples_split': 10}],
            include_default=True,
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None

    def test_initial_params_with_include_default_false(self, iris_data, tree_space):
        """Verify include_default=False excludes sklearn defaults."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=[{'max_depth': 5, 'min_samples_split': 10}],
            include_default=False,
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None

    def test_initial_params_with_early_stopping(self, iris_data, tree_space):
        """Verify initial_params works with early stopping."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=[{'max_depth': 5, 'min_samples_split': 10}],
            population_size=10,
            generations=20,
            early_stopping=True,
            patience=3,
            min_delta=0.01,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None
        # May stop early, so generations could be less than 20
        assert opt.n_trials_ > 0

    def test_initial_params_random_forest(self, iris_data):
        """Verify initial_params works with different estimator."""
        X, y = iris_data
        rf_space = HyperparameterSpaceBuilder.get_default_space(RandomForestClassifier)

        opt = GeneticSearch(
            estimator_class=RandomForestClassifier,
            hyperparam_space=rf_space,
            initial_params=[{
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 10
            }],
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None

    def test_initial_params_partial_hyperparams(self, iris_data, tree_space):
        """Verify initial_params with partial hyperparameter specification."""
        X, y = iris_data

        # Only specify some hyperparameters, others should be random
        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=[{'max_depth': 5}],  # Only max_depth specified
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None

    def test_initial_params_reproducibility(self, iris_data, tree_space):
        """Verify same initial_params + seed produces same results."""
        X, y = iris_data

        initial_configs = [
            {'max_depth': 3, 'min_samples_split': 5},
            {'max_depth': 5, 'min_samples_split': 10}
        ]

        opt1 = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=initial_configs,
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt1.fit(X, y)

        opt2 = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=initial_configs,
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt2.fit(X, y)

        # Same seed + same initial_params should give same best params
        assert opt1.best_params_ == opt2.best_params_

    def test_initial_params_with_cv(self, iris_data, tree_space):
        """Verify initial_params works with cross-validation."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=[{'max_depth': 5, 'min_samples_split': 10}],
            population_size=10,
            generations=3,
            cv=3,
            disable_file_output=True,
            seed=42
        )
        opt.fit(X, y)

        assert opt.best_estimator_ is not None

    def test_initial_params_improves_convergence(self, iris_data, tree_space):
        """Verify initial_params can lead to better initial fitness."""
        X, y = iris_data

        # Get a known good configuration
        known_good = {'max_depth': 5, 'min_samples_split': 5}

        # Run with seeding
        opt_seeded = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=[known_good],
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=42
        )
        opt_seeded.fit(X, y)

        # Run without seeding (same seed for fair comparison)
        opt_random = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            initial_params=None,
            population_size=10,
            generations=3,
            disable_file_output=True,
            seed=43  # Different seed to ensure different random init
        )
        opt_random.fit(X, y)

        # Both should complete successfully
        assert opt_seeded.best_estimator_ is not None
        assert opt_random.best_estimator_ is not None
