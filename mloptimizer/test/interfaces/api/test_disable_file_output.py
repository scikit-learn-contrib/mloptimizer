"""
Tests for disable_file_output feature.

This module tests the disable_file_output parameter which controls whether
GeneticSearch creates output directories and files during optimization.
"""

import os
import tempfile
import pytest
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder


class TestDisableFileOutput:
    """Test suite for disable_file_output parameter."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset for testing."""
        return load_iris(return_X_y=True)

    @pytest.fixture
    def tree_space(self):
        """Get default hyperparameter space for DecisionTreeClassifier."""
        return HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

    def test_disable_file_output_true_default(self, iris_data, tree_space):
        """Verify disable_file_output=True (default) creates no files."""
        X, y = iris_data

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)

                opt = GeneticSearch(
                    estimator_class=DecisionTreeClassifier,
                    hyperparam_space=tree_space,
                    disable_file_output=True,  # Explicit default
                    generations=3,
                    population_size=10,
                    seed=42
                )
                opt.fit(X, y)

                # Verify no directories created (no timestamped directories)
                dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
                assert len(dirs) == 0, f"Expected no directories, found: {dirs}"

                # Verify no files created
                files = [f for f in os.listdir('.') if os.path.isfile(f)]
                assert len(files) == 0, f"Expected no files, found: {files}"

            finally:
                os.chdir(original_dir)

    def test_disable_file_output_implicit_default(self, iris_data, tree_space):
        """Verify default behavior (no parameter) creates no files."""
        X, y = iris_data

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Don't specify disable_file_output - should default to True
                opt = GeneticSearch(
                    estimator_class=DecisionTreeClassifier,
                    hyperparam_space=tree_space,
                    generations=3,
                    population_size=10,
                    seed=42
                )
                opt.fit(X, y)

                # Verify no output created
                items = os.listdir('.')
                assert len(items) == 0, f"Expected no output, found: {items}"

            finally:
                os.chdir(original_dir)

    def test_disable_file_output_false_creates_files(self, iris_data, tree_space):
        """Verify disable_file_output=False creates output directory."""
        X, y = iris_data

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)

                opt = GeneticSearch(
                    estimator_class=DecisionTreeClassifier,
                    hyperparam_space=tree_space,
                    disable_file_output=False,  # Enable file output
                    generations=3,
                    population_size=10,
                    seed=42
                )
                opt.fit(X, y)

                # Should create timestamped directory
                dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
                assert len(dirs) > 0, "Expected output directory to be created"

                # Directory name should match pattern: YYYYMMDD_HHMMSS_EstimatorName
                assert any('DecisionTreeClassifier' in d for d in dirs), \
                    f"Expected DecisionTreeClassifier directory, found: {dirs}"

            finally:
                os.chdir(original_dir)

    def test_disable_file_output_with_early_stopping(self, iris_data, tree_space):
        """Verify disable_file_output=True works with early stopping."""
        X, y = iris_data

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)

                opt = GeneticSearch(
                    estimator_class=DecisionTreeClassifier,
                    hyperparam_space=tree_space,
                    disable_file_output=True,
                    early_stopping=True,
                    patience=3,
                    min_delta=0.01,
                    generations=20,
                    population_size=10,
                    seed=42
                )
                opt.fit(X, y)

                # No files should be created even with early stopping
                items = os.listdir('.')
                assert len(items) == 0, f"Expected no output, found: {items}"

            finally:
                os.chdir(original_dir)

    def test_disable_file_output_with_initial_params(self, iris_data, tree_space):
        """Verify disable_file_output=True works with initial_params."""
        X, y = iris_data

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)

                opt = GeneticSearch(
                    estimator_class=DecisionTreeClassifier,
                    hyperparam_space=tree_space,
                    disable_file_output=True,
                    initial_params=[{'max_depth': 5, 'min_samples_split': 10}],
                    generations=3,
                    population_size=10,
                    seed=42
                )
                opt.fit(X, y)

                # No files should be created
                items = os.listdir('.')
                assert len(items) == 0, f"Expected no output, found: {items}"

            finally:
                os.chdir(original_dir)

    def test_disable_file_output_optimization_still_works(self, iris_data, tree_space):
        """Verify optimization works correctly regardless of file output setting."""
        X, y = iris_data

        # Run with file output disabled
        opt_no_files = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            disable_file_output=True,
            generations=3,
            population_size=10,
            seed=42
        )
        opt_no_files.fit(X, y)

        # Run with file output enabled
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)

                opt_with_files = GeneticSearch(
                    estimator_class=DecisionTreeClassifier,
                    hyperparam_space=tree_space,
                    disable_file_output=False,
                    generations=3,
                    population_size=10,
                    seed=42
                )
                opt_with_files.fit(X, y)

            finally:
                os.chdir(original_dir)

        # Both should produce valid results
        assert opt_no_files.best_estimator_ is not None
        assert opt_with_files.best_estimator_ is not None

        # Same seed should give same results
        assert opt_no_files.best_params_ == opt_with_files.best_params_

    def test_disable_file_output_attributes_available(self, iris_data, tree_space):
        """Verify performance tracking attributes work with disable_file_output=True."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            disable_file_output=True,
            generations=3,
            population_size=10,
            seed=42
        )
        opt.fit(X, y)

        # These attributes should be available even without file output
        assert hasattr(opt, 'n_trials_')
        assert hasattr(opt, 'optimization_time_')
        assert hasattr(opt, 'best_estimator_')
        assert hasattr(opt, 'best_params_')
        assert hasattr(opt, 'cv_results_')
        assert hasattr(opt, 'logbook_')
        assert hasattr(opt, 'populations_')

        # Verify they have valid values
        assert opt.n_trials_ > 0
        assert opt.optimization_time_ > 0

    def test_disable_file_output_with_cv(self, iris_data, tree_space):
        """Verify disable_file_output=True works with cross-validation."""
        X, y = iris_data

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)

                opt = GeneticSearch(
                    estimator_class=DecisionTreeClassifier,
                    hyperparam_space=tree_space,
                    disable_file_output=True,
                    cv=3,
                    generations=3,
                    population_size=10,
                    seed=42
                )
                opt.fit(X, y)

                # No files should be created
                items = os.listdir('.')
                assert len(items) == 0, f"Expected no output, found: {items}"

            finally:
                os.chdir(original_dir)

    def test_disable_file_output_parallel_execution(self, iris_data, tree_space):
        """Verify disable_file_output=True works with parallel execution."""
        X, y = iris_data

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)

                opt = GeneticSearch(
                    estimator_class=DecisionTreeClassifier,
                    hyperparam_space=tree_space,
                    disable_file_output=True,
                    use_parallel=True,
                    generations=3,
                    population_size=10,
                    seed=42
                )
                opt.fit(X, y)

                # No files should be created even with parallelization
                items = os.listdir('.')
                assert len(items) == 0, f"Expected no output, found: {items}"

            finally:
                os.chdir(original_dir)

    def test_disable_file_output_logbook_populated(self, iris_data, tree_space):
        """Verify logbook is populated even with disable_file_output=True."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            disable_file_output=True,
            generations=5,
            population_size=10,
            seed=42
        )
        opt.fit(X, y)

        # Logbook should be populated
        assert opt.logbook_ is not None
        assert len(opt.logbook_) > 0

        # Check logbook has expected statistics
        first_record = opt.logbook_[0]
        assert 'gen' in first_record
        assert 'nevals' in first_record
        assert 'avg' in first_record or 'min' in first_record or 'max' in first_record

    def test_disable_file_output_populations_dataframe_available(self, iris_data, tree_space):
        """Verify populations DataFrame is available with disable_file_output=True."""
        X, y = iris_data

        opt = GeneticSearch(
            estimator_class=DecisionTreeClassifier,
            hyperparam_space=tree_space,
            disable_file_output=True,
            generations=3,
            population_size=10,
            seed=42
        )
        opt.fit(X, y)

        # Populations DataFrame should be available
        assert opt.populations_ is not None
        assert len(opt.populations_) > 0

        # Should have columns for hyperparameters and fitness
        assert 'fitness' in opt.populations_.columns
