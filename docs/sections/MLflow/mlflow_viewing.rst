Viewing MLflow Results
======================

After running optimizations with MLflow enabled, you can view and analyze results using several methods.

MLflow UI (Recommended)
-----------------------

The MLflow UI provides an interactive web interface to explore your optimization runs.

Starting the UI
~~~~~~~~~~~~~~~

From your project directory (where ``mlruns/`` exists):

.. code-block:: bash

   mlflow ui --port 5000

Then open your browser to http://localhost:5000

If using a custom tracking location:

.. code-block:: bash

   # For database backend
   mlflow ui --backend-store-uri sqlite:///path/to/mlflow.db --port 5000

   # For file-based backend
   mlflow ui --backend-store-uri file:///path/to/mlruns --port 5000

UI Features
~~~~~~~~~~~

The MLflow UI allows you to:

**Browse Experiments**
   View all experiments and their runs in a table format

**Compare Runs**
   Select multiple runs to compare metrics and hyperparameters side-by-side

**View Generation Evolution**
   Plot generation-level metrics to visualize fitness evolution over time

**Filter and Search**
   Use powerful filters to find runs based on parameters, metrics, or tags

**Sort by Performance**
   Order runs by best fitness, total evaluations, or optimization time

**Download Artifacts**
   Access any artifacts logged during optimization

Interpreting Run Names
~~~~~~~~~~~~~~~~~~~~~~~

mloptimizer creates runs with descriptive names:

- **Parent runs**: ``YYYYMMDD_HHMMSS_EstimatorName``

  Example: ``20260118_161428_RandomForestClassifier``

- **Child runs** (individual evaluations): ``gen_N_ind_M_EstimatorName``

  Example: ``gen_3_ind_5_RandomForestClassifier``

Parent runs contain generation-level metrics and overall optimization results. Child runs contain individual evaluation details.

Using the Python API
--------------------

Query MLflow programmatically for custom analysis:

Basic Run Retrieval
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import mlflow
   from mlflow.tracking import MlflowClient

   # Get experiment
   experiment = mlflow.get_experiment_by_name("mloptimizer")

   # Search all runs
   all_runs = mlflow.search_runs(
       experiment_ids=[experiment.experiment_id],
       order_by=["start_time DESC"]
   )

   print(f"Found {len(all_runs)} runs")
   print(all_runs[['run_id', 'start_time', 'metrics.final_best_fitness']])

Getting Parent Runs Only
~~~~~~~~~~~~~~~~~~~~~~~~~

To filter for parent runs (optimization-level):

.. code-block:: python

   import re

   # Parent runs have timestamp-based names
   parent_runs = [
       r for r in all_runs.itertuples()
       if re.match(r'^\d{8}_\d{6}_', getattr(r, 'tags.mlflow.runName', ''))
   ]

   print(f"Found {len(parent_runs)} optimization runs")

Retrieving Generation History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get generation-level metrics for a specific run:

.. code-block:: python

   from mlflow.tracking import MlflowClient

   client = MlflowClient()
   run_id = "your_run_id_here"  # From parent run

   # Get generation best fitness history
   gen_best = client.get_metric_history(run_id, "generation_best_fitness")
   gen_avg = client.get_metric_history(run_id, "generation_avg_fitness")

   # Print evolution
   for metric in gen_best:
       print(f"Generation {metric.step}: Best={metric.value:.4f}")

Plotting Evolution
~~~~~~~~~~~~~~~~~~

Visualize fitness evolution across generations:

.. code-block:: python

   import matplotlib.pyplot as plt
   from mlflow.tracking import MlflowClient

   client = MlflowClient()
   run_id = "your_run_id_here"

   # Get metric history
   gen_best = client.get_metric_history(run_id, "generation_best_fitness")
   gen_avg = client.get_metric_history(run_id, "generation_avg_fitness")
   gen_worst = client.get_metric_history(run_id, "generation_worst_fitness")

   # Extract data
   generations = [m.step for m in gen_best]
   best_fitness = [m.value for m in gen_best]
   avg_fitness = [m.value for m in gen_avg]
   worst_fitness = [m.value for m in gen_worst]

   # Create plot
   plt.figure(figsize=(10, 6))
   plt.plot(generations, best_fitness, 'g-o', label='Best', linewidth=2)
   plt.plot(generations, avg_fitness, 'b-s', label='Average', linewidth=2)
   plt.plot(generations, worst_fitness, 'r-^', label='Worst', linewidth=2)
   plt.xlabel('Generation', fontsize=12)
   plt.ylabel('Fitness', fontsize=12)
   plt.title('Evolution of Population Fitness', fontsize=14)
   plt.legend(fontsize=10)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('evolution.png', dpi=150)
   plt.show()

Comparing Multiple Runs
~~~~~~~~~~~~~~~~~~~~~~~~

Compare hyperparameters and results across runs:

.. code-block:: python

   import mlflow
   import pandas as pd

   # Get all runs from experiment
   runs = mlflow.search_runs(experiment_ids=["1"])

   # Select comparison columns
   comparison = runs[[
       'run_id',
       'metrics.final_best_fitness',
       'params.population_size',
       'params.generations',
       'tags.estimator_class',
       'tags.early_stopped',
       'tags.optimization_time_seconds'
   ]].copy()

   # Sort by performance
   comparison = comparison.sort_values(
       'metrics.final_best_fitness',
       ascending=False
   )

   print(comparison)

Using the Extraction Script
----------------------------

mloptimizer provides a ready-to-use extraction script for comprehensive analysis:

.. code-block:: bash

   python scripts/mlflow_extract.py

This script automatically:

1. Finds all MLflow experiments
2. Extracts runs with metrics and parameters
3. Analyzes generation evolution
4. Exports results to CSV files:

   - ``mlflow_all_runs.csv`` - Complete run data
   - ``mlflow_runs_simple.csv`` - Key metrics and parameters
   - ``mlflow_hyperparameter_analysis.csv`` - Hyperparameter statistics
   - ``mlflow_generation_summary.csv`` - Evolution across generations

Example output:

.. code-block:: text

   ================================================================================
   MLFLOW RESULTS EXTRACTION
   ================================================================================

   MLflow tracking URI: file:./mlruns

   --------------------------------------------------------------------------------
   AVAILABLE EXPERIMENTS
   --------------------------------------------------------------------------------

   Found 2 experiments:
     - Experiment ID: 0, Name: Default
     - Experiment ID: 1, Name: mloptimizer

   Analyzing experiment: mloptimizer (ID: 1)

   ================================================================================
   BEST RUN ANALYSIS
   ================================================================================

   Best run ID: a7f3c9e8d4b5...
   Start time: 2026-01-18 16:21:24
   Status: FINISHED
   Duration: 18.72 seconds

   ----------------------------------------
   METRICS
   ----------------------------------------
     final_best_fitness: 0.965675
     generation_avg_fitness: 0.960300
     generation_best_fitness: 0.965675

   ----------------------------------------
   HYPERPARAMETERS
   ----------------------------------------
     generations: 10
     population_size: 20
     early_stopping: True
     patience: 3

   ================================================================================
   EVOLUTION ANALYSIS (ACROSS GENERATIONS)
   ================================================================================

   Generation  Best      Average   Worst     StdDev
   0          0.9615    0.9557    0.9498    0.0047
   1          0.9615    0.9557    0.9498    0.0047
   2          0.9625    0.9590    0.9555    0.0028
   3          0.9629    0.9578    0.9527    0.0041
   4          0.9657    0.9603    0.9549    0.0043

Analyzing Tags and Metadata
----------------------------

Retrieve and analyze run metadata:

.. code-block:: python

   import mlflow

   # Get a specific run
   run = mlflow.get_run(run_id)

   # Access tags
   print("Dataset Information:")
   print(f"  Samples: {run.data.tags.get('dataset_samples')}")
   print(f"  Features: {run.data.tags.get('dataset_features')}")
   print(f"  Classes: {run.data.tags.get('dataset_classes')}")

   print("\nOptimization Results:")
   print(f"  Estimator: {run.data.tags.get('estimator_class')}")
   print(f"  Early stopped: {run.data.tags.get('early_stopped')}")
   print(f"  Total evaluations: {run.data.tags.get('total_evaluations')}")
   print(f"  Time: {run.data.tags.get('optimization_time_seconds')}s")

   # Access parameters
   print("\nConfiguration:")
   print(f"  Population: {run.data.params.get('population_size')}")
   print(f"  Generations: {run.data.params.get('generations')}")
   print(f"  Parallel: {run.data.params.get('use_parallel')}")

Exporting Data for External Analysis
-------------------------------------

Export MLflow data for use in other tools:

.. code-block:: python

   import mlflow
   import pandas as pd

   # Get all runs
   runs = mlflow.search_runs(experiment_ids=["1"])

   # Export to CSV
   runs.to_csv('mlflow_results.csv', index=False)

   # Export to Excel
   runs.to_excel('mlflow_results.xlsx', index=False)

   # Export to JSON
   runs.to_json('mlflow_results.json', orient='records')

.. tip::
   The MLflow UI is the fastest way to get an overview of your experiments. Use the Python API for custom analysis and automated reporting.

.. note::
   Generation-level metrics use the ``step`` parameter in MLflow, allowing you to visualize evolution as a time series in the MLflow UI.
