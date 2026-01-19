Logging Configuration
=====================

mloptimizer follows the standard Python library logging pattern, giving you full control over log output. By default, the library uses a `NullHandler`, meaning no log messages are displayed unless you explicitly configure logging.

Using the verbose Parameter (Recommended)
-----------------------------------------

The easiest way to enable logging is using the ``verbose`` parameter in ``GeneticSearch``:

.. code-block:: python

    from mloptimizer.interfaces import GeneticSearch

    # Silent (default)
    opt = GeneticSearch(estimator_class=..., hyperparam_space=..., verbose=0)

    # Info level - shows optimization lifecycle
    opt = GeneticSearch(estimator_class=..., hyperparam_space=..., verbose=1)

    # Debug level - shows detailed evaluation info
    opt = GeneticSearch(estimator_class=..., hyperparam_space=..., verbose=2)

This is similar to how scikit-learn and XGBoost handle verbosity.

Basic Configuration
-------------------

To enable logging output, configure Python's logging module before running your optimization:

.. code-block:: python

    import logging

    # Enable INFO level logging globally
    logging.basicConfig(level=logging.INFO)

    # Now use mloptimizer
    from mloptimizer.interfaces import GeneticSearch
    # ... your optimization code

This will display informative messages about the optimization process, including start/end summaries and progress updates.

Configuring mloptimizer's Logger Only
-------------------------------------

To enable logging specifically for mloptimizer without affecting other libraries:

.. code-block:: python

    import logging

    # Get mloptimizer's logger
    mlopt_logger = logging.getLogger("mloptimizer")
    mlopt_logger.setLevel(logging.INFO)

    # Add a handler to output to console
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
    mlopt_logger.addHandler(handler)

Logging to a File
-----------------

To save logs to a file for later analysis:

.. code-block:: python

    import logging

    # Configure logging to file
    logging.basicConfig(
        filename='optimization.log',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s"
    )

Or configure both console and file output:

.. code-block:: python

    import logging

    # Create logger
    logger = logging.getLogger("mloptimizer")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler("optimization.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
    logger.addHandler(file_handler)

Log Levels
----------

mloptimizer uses standard Python logging levels:

- **DEBUG**: Detailed information for diagnosing problems (individual evaluations, internal state)
- **INFO**: Confirmation that things are working as expected (optimization start/end, generation summaries)
- **WARNING**: Indication of potential issues (deprecated parameters, suboptimal configurations)
- **ERROR**: Serious problems that prevent operation

Example with DEBUG level:

.. code-block:: python

    import logging
    logging.getLogger("mloptimizer").setLevel(logging.DEBUG)

Silencing Logs
--------------

To completely suppress mloptimizer's log output:

.. code-block:: python

    import logging
    logging.getLogger("mloptimizer").setLevel(logging.CRITICAL)

Or to only see warnings and errors:

.. code-block:: python

    import logging
    logging.getLogger("mloptimizer").setLevel(logging.WARNING)

Integration with scikit-learn
-----------------------------

Since mloptimizer follows the same logging pattern as scikit-learn and other major Python libraries, you can configure logging for your entire ML pipeline consistently:

.. code-block:: python

    import logging

    # Configure logging for all ML libraries
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    # Now all libraries (mloptimizer, sklearn, etc.) will log consistently
