Basic MLflow Usage
==================

Getting Started with MLflow
----------------------------

MLflow tracking is easily enabled in mloptimizer by setting the ``use_mlflow`` parameter to ``True`` when creating a ``GeneticSearch`` instance.

Installation
~~~~~~~~~~~~

First, ensure MLflow is installed:

.. code-block:: bash

   pip install mlflow

If you attempt to use ``use_mlflow=True`` without MLflow installed, mloptimizer will display a clear error message with installation instructions.

Enabling MLflow Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~

Simply add ``use_mlflow=True`` to your ``GeneticSearch`` configuration:

.. code-block:: python

   from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import StratifiedKFold

   # Load data
   X, y = load_breast_cancer(return_X_y=True)

   # Get hyperparameter space
   space = HyperparameterSpaceBuilder.get_default_space(RandomForestClassifier)

   # Create GeneticSearch with MLflow enabled
   opt = GeneticSearch(
       estimator_class=RandomForestClassifier,
       hyperparam_space=space,
       cv=StratifiedKFold(n_splits=5),
       scoring="balanced_accuracy",
       generations=10,
       population_size=20,
       early_stopping=True,
       patience=3,
       use_mlflow=True  # Enable MLflow tracking
   )

   # Run optimization - all results logged to MLflow
   opt.fit(X, y)

   print(f"Best model: {opt.best_estimator_}")
   print("\nView results in MLflow UI:")
   print("  mlflow ui --port 5000")
   print("  Then open: http://localhost:5000")

What Gets Logged
----------------

When MLflow tracking is enabled, mloptimizer automatically logs comprehensive information about your optimization run:

Parent Run (Optimization-Level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each optimization creates a parent run with a timestamp-based name (e.g., ``20260118_161428_RandomForestClassifier``):

**Generation-Level Metrics**
   - ``generation_best_fitness``: Best fitness in each generation
   - ``generation_avg_fitness``: Average fitness in each generation
   - ``generation_worst_fitness``: Worst fitness in each generation
   - ``generation_median_fitness``: Median fitness in each generation
   - ``final_best_fitness``: Best fitness achieved overall

**Configuration Parameters**
   - ``population_size``: Population size
   - ``generations``: Maximum generations
   - ``early_stopping``: Whether early stopping is enabled
   - ``patience``: Early stopping patience
   - ``use_parallel``: Parallelization status
   - ``n_evolvable_params``: Number of evolvable hyperparameters
   - ``evolvable_params``: List of evolvable parameter names

**Dataset Metadata (Tags)**
   - ``dataset_samples``: Number of training samples
   - ``dataset_features``: Number of features
   - ``dataset_classes``: Number of classes (or 'regression')

**Optimization Results (Tags)**
   - ``estimator_class``: Name of the estimator being optimized
   - ``early_stopped``: Whether optimization stopped early
   - ``stopped_at_generation``: Generation where optimization stopped
   - ``total_evaluations``: Total number of model evaluations
   - ``optimization_time_seconds``: Total optimization time

Child Runs (Individual Evaluations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each individual evaluation creates a nested child run:

- **Hyperparameters**: All hyperparameter values for that individual
- **Fitness Metrics**: Evaluation scores (accuracy, balanced_accuracy, etc.)
- **Generation Info**: Tags indicating which generation and individual index

Default Storage Location
-------------------------

By default, MLflow stores runs locally in the ``./mlruns/`` directory:

.. code-block:: text

   ./mlruns/
   ├── 0/                          # Default experiment
   ├── 1/                          # mloptimizer experiment
   │   ├── meta.yaml
   │   └── <run_id>/
   │       ├── meta.yaml
   │       ├── metrics/
   │       │   ├── generation_best_fitness
   │       │   ├── generation_avg_fitness
   │       │   └── final_best_fitness
   │       ├── params/
   │       │   ├── population_size
   │       │   ├── generations
   │       │   └── ...
   │       └── tags/
   │           ├── estimator_class
   │           ├── dataset_samples
   │           └── ...

Custom Experiment Name
----------------------

You can specify a custom experiment name using the MLflow API before creating your ``GeneticSearch``:

.. code-block:: python

   import mlflow
   from mloptimizer.interfaces import GeneticSearch

   # Set custom experiment name
   mlflow.set_experiment("breast_cancer_optimization")

   # Create GeneticSearch - will log to this experiment
   opt = GeneticSearch(
       estimator_class=YourEstimator,
       hyperparam_space=space,
       use_mlflow=True
   )

   opt.fit(X, y)

Example with Early Stopping
----------------------------

MLflow tracking captures early stopping information:

.. code-block:: python

   from mloptimizer.interfaces import GeneticSearch
   from sklearn.ensemble import GradientBoostingClassifier

   opt = GeneticSearch(
       estimator_class=GradientBoostingClassifier,
       hyperparam_space=space,
       cv=cv,
       scoring="balanced_accuracy",
       generations=20,
       population_size=30,
       early_stopping=True,  # Enable early stopping
       patience=5,
       min_delta=0.001,
       use_mlflow=True  # Track everything
   )

   opt.fit(X, y)

The MLflow run will include:
- Tag indicating early stopping was enabled
- Tag showing which generation optimization stopped at
- Tag showing the reason (no improvement for N generations)
- Final best fitness achieved

Disabling MLflow
-----------------

MLflow tracking is disabled by default. To explicitly disable it:

.. code-block:: python

   opt = GeneticSearch(
       estimator_class=YourEstimator,
       hyperparam_space=space,
       use_mlflow=False  # Disable MLflow (default)
   )

When disabled, no MLflow data is logged, and the library works without requiring MLflow to be installed.

.. note::
   MLflow tracking adds minimal overhead to optimization time. The benefits of comprehensive experiment tracking typically outweigh the small performance cost.

.. tip::
   For quick experiments, use local MLflow storage. For production or team collaboration, configure a remote MLflow tracking server (see :doc:`mlflow_remote`).
