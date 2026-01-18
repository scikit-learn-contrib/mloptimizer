Remote MLflow Tracking
======================

While local MLflow storage is convenient for individual work, remote MLflow servers enable team collaboration, centralized experiment management, and production deployments.

Overview
--------

By default, mloptimizer uses local file-based MLflow tracking. However, you can configure it to use remote MLflow tracking servers by setting the tracking URI **before** creating your ``GeneticSearch`` instance.

mloptimizer fully supports remote MLflow servers - no special configuration is needed beyond setting the tracking URI using the MLflow API.

Configuration Methods
---------------------

Method 1: Python API (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure the tracking URI in your code before creating ``GeneticSearch``:

.. code-block:: python

   import mlflow
   from mloptimizer.interfaces import GeneticSearch

   # Configure remote MLflow server
   mlflow.set_tracking_uri("http://mlflow-server.company.com:5000")

   # Optional: Set experiment name
   mlflow.set_experiment("production_optimization")

   # Create GeneticSearch - will log to remote server
   opt = GeneticSearch(
       estimator_class=YourEstimator,
       hyperparam_space=space,
       use_mlflow=True  # Logs to configured remote server
   )

   opt.fit(X, y)

Method 2: Environment Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the ``MLFLOW_TRACKING_URI`` environment variable:

.. code-block:: bash

   export MLFLOW_TRACKING_URI=http://mlflow-server.company.com:5000
   python your_optimization_script.py

.. code-block:: python

   # In your script - no tracking URI configuration needed
   from mloptimizer.interfaces import GeneticSearch

   # Will automatically use MLFLOW_TRACKING_URI
   opt = GeneticSearch(..., use_mlflow=True)
   opt.fit(X, y)

Configuration Priority
~~~~~~~~~~~~~~~~~~~~~~

MLflow uses this priority order:

1. Explicit ``mlflow.set_tracking_uri()`` call (highest priority)
2. ``MLFLOW_TRACKING_URI`` environment variable
3. Default local file-based (``./mlruns/``)

Starting a Local MLflow Server
-------------------------------

You can run MLflow in server mode on your local machine for testing remote configuration:

Basic Local Server
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mlflow server --host 127.0.0.1 --port 5000

With Database Backend
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mlflow server \
       --backend-store-uri sqlite:///mlflow.db \
       --default-artifact-root ./mlruns \
       --host 127.0.0.1 \
       --port 5000

Then configure your code to use it:

.. code-block:: python

   import mlflow
   mlflow.set_tracking_uri("http://127.0.0.1:5000")

Production MLflow Server Setup
-------------------------------

For production environments, use a robust database backend and remote artifact storage.

PostgreSQL Backend with S3 Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mlflow server \
       --backend-store-uri postgresql://user:password@host:5432/mlflow_db \
       --default-artifact-root s3://your-bucket/mlflow-artifacts \
       --host 0.0.0.0 \
       --port 5000

Then in your code:

.. code-block:: python

   import mlflow
   import os

   # Configure AWS credentials (if needed)
   os.environ['AWS_ACCESS_KEY_ID'] = 'your-key'
   os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret'

   # Point to production server
   mlflow.set_tracking_uri("http://mlflow-prod.company.com:5000")

   # Run optimization
   opt = GeneticSearch(..., use_mlflow=True)
   opt.fit(X, y)

MySQL Backend with Azure Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mlflow server \
       --backend-store-uri mysql://user:password@host:3306/mlflow_db \
       --default-artifact-root wasbs://container@account.blob.core.windows.net/mlflow \
       --host 0.0.0.0 \
       --port 5000

Cloud-Based MLflow
------------------

Databricks MLflow
~~~~~~~~~~~~~~~~~

Databricks provides managed MLflow hosting:

.. code-block:: python

   import mlflow

   # Configure Databricks MLflow
   mlflow.set_tracking_uri("databricks")
   mlflow.set_experiment("/Users/your-name/model-optimization")

   # Requires databricks-cli configured
   opt = GeneticSearch(..., use_mlflow=True)
   opt.fit(X, y)

AWS Managed MLflow
~~~~~~~~~~~~~~~~~~

If using AWS SageMaker with MLflow:

.. code-block:: python

   import mlflow

   mlflow.set_tracking_uri("https://your-mlflow-endpoint.amazonaws.com")
   opt = GeneticSearch(..., use_mlflow=True)
   opt.fit(X, y)

Team Collaboration Example
---------------------------

Setting up MLflow for team collaboration:

Server Setup (DevOps)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # On dedicated MLflow server
   mlflow server \
       --backend-store-uri postgresql://mlflow:password@db.company.com:5432/mlflow \
       --default-artifact-root s3://company-mlflow/artifacts \
       --host 0.0.0.0 \
       --port 80

Team Member Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each team member configures the same tracking URI:

.. code-block:: python

   import mlflow
   from mloptimizer.interfaces import GeneticSearch

   # Point to shared server
   mlflow.set_tracking_uri("http://mlflow.company.com")

   # Use team experiment
   mlflow.set_experiment("team_model_optimization")

   # Run optimization - visible to whole team
   opt = GeneticSearch(
       estimator_class=RandomForestClassifier,
       hyperparam_space=space,
       use_mlflow=True
   )

   opt.fit(X, y)

All runs are logged to the central server and visible to the entire team via the MLflow UI at http://mlflow.company.com

Viewing Results from Remote Server
-----------------------------------

MLflow UI from Remote Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the MLflow server has a web UI (it does by default):

.. code-block:: bash

   # Just open in browser
   http://mlflow-server.company.com:5000

Or configure a local MLflow UI to connect to the remote server:

.. code-block:: bash

   mlflow ui --backend-store-uri http://mlflow-server.company.com:5000

Python API with Remote Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import mlflow

   # Configure remote server
   mlflow.set_tracking_uri("http://mlflow-server.company.com:5000")

   # Query runs from remote server
   runs = mlflow.search_runs(experiment_ids=["1"])
   print(f"Found {len(runs)} runs on remote server")

Verification Test
-----------------

The existing MLflow test demonstrates remote server usage:

.. code-block:: python

   # From mloptimizer/test/interfaces/api/test_genetic_search_mlflow_tracking.py

   MLFLOW_PORT = 5001
   MLFLOW_TRACKING_URI = f"http://127.0.0.1:{MLFLOW_PORT}"

   def test_genetic_search_creates_mlflow_runs(mlflow_server):
       # Configure remote server
       mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

       # Run optimization - logs to remote server
       search = GeneticSearch(
           estimator_class=DecisionTreeClassifier,
           hyperparam_space=space,
           use_mlflow=True
       )
       search.fit(X_train, y_train)

       # Verify runs logged to remote server
       client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
       runs = client.search_runs([experiment.experiment_id])
       assert len(runs) > 0  # Runs found on remote server

Run this test to verify remote MLflow works:

.. code-block:: bash

   pytest mloptimizer/test/interfaces/api/test_genetic_search_mlflow_tracking.py -v

Security Considerations
-----------------------

When using remote MLflow servers:

Authentication
~~~~~~~~~~~~~~

For production servers, enable authentication:

.. code-block:: bash

   mlflow server \
       --backend-store-uri postgresql://... \
       --default-artifact-root s3://... \
       --host 0.0.0.0 \
       --port 5000 \
       --app-name basic-auth

Configure credentials:

.. code-block:: python

   import os

   os.environ['MLFLOW_TRACKING_USERNAME'] = 'your-username'
   os.environ['MLFLOW_TRACKING_PASSWORD'] = 'your-password'

   mlflow.set_tracking_uri("http://mlflow-server.company.com:5000")

HTTPS
~~~~~

Use HTTPS for production:

.. code-block:: python

   mlflow.set_tracking_uri("https://mlflow-server.company.com")

Network Security
~~~~~~~~~~~~~~~~

- Configure firewalls to restrict MLflow server access
- Use VPN for accessing internal MLflow servers
- Implement proper database access controls

Troubleshooting
---------------

Cannot Connect to Server
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test connection
   import mlflow
   mlflow.set_tracking_uri("http://mlflow-server.company.com:5000")

   try:
       experiments = mlflow.search_experiments()
       print(f"✓ Connected successfully. Found {len(experiments)} experiments.")
   except Exception as e:
       print(f"✗ Connection failed: {e}")

**Common issues:**

- Server not running: Check ``mlflow server`` process
- Firewall: Verify port is open
- Wrong URL: Check hostname and port
- Network: Verify connectivity with ``ping`` or ``curl``

Different Results on Different Machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure all team members use the same tracking URI:

.. code-block:: python

   import mlflow
   print(f"Current tracking URI: {mlflow.get_tracking_uri()}")

Slow Performance
~~~~~~~~~~~~~~~~

For large artifacts, use appropriate storage:

- Local server: Use SSD storage
- Remote server: Use S3/Azure/GCS for artifacts
- Database: Use PostgreSQL/MySQL instead of SQLite

.. tip::
   Start with a local MLflow server for development, then migrate to a production server with database backend and cloud storage as your needs grow.

.. note::
   Remote MLflow servers require network connectivity. Ensure your optimization runs can reach the server, or use local fallback for offline development.

.. warning::
   Storing credentials in code is insecure. Use environment variables or secure secret management for production deployments.
