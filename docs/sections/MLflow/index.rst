MLflow Integration
==================

MLflow is an open-source platform for managing the machine learning lifecycle, including experiment tracking, model versioning, and deployment. The `mloptimizer` library integrates seamlessly with MLflow to provide comprehensive tracking of genetic algorithm optimization runs, enabling you to monitor evolution progress, compare hyperparameter configurations, and analyze results.

.. toctree::
   :hidden:

   mlflow_basics
   mlflow_viewing
   mlflow_remote

Overview of MLflow Features
----------------------------

- **Experiment Tracking**: Automatically log all optimization runs with their configurations, metrics, and results. Track generation-level metrics to visualize how fitness evolves across generations.

- **Result Visualization**: Use the MLflow UI to interactively explore runs, compare different optimization strategies, and analyze hyperparameter impact on model performance.

- **Remote Tracking**: Configure MLflow to use remote tracking servers for team collaboration and centralized experiment management. Share optimization results across your organization.

Each section provides detailed guidance on using MLflow with mloptimizer.

Key Benefits
------------

**Generation-Level Tracking**
   Every generation's best, average, and worst fitness scores are logged, allowing you to visualize the evolution of your population over time.

**Comprehensive Metadata**
   Dataset characteristics, optimization configuration, early stopping information, and timing metrics are automatically recorded.

**Flexible Storage**
   Use local file-based storage for quick experiments or configure remote MLflow servers with database backends for production deployments.

**Seamless Integration**
   Simply add ``use_mlflow=True`` to your ``GeneticSearch`` configuration - no additional code required.
