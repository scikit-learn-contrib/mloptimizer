=============================
Optimizer Directory Structure
=============================

When an optimizer is run, it generates a directory within the current working directory (or the specified output directory) to store all optimization results. This directory, named in the format `YYYYMMDD_nnnnnnnnnn_OptimizerName`, organizes the outputs from the optimization process, including the best estimator, checkpoints, visualizations, logs, and progress data.

Directory Structure
-------------------

The directory structure is organized as follows:

.. code-block:: bash

    ├── checkpoints
    │   ├── cp_gen_0.pkl
    │   └── cp_gen_1.pkl
    ├── graphics
    │   ├── logbook.html
    │   └── search_space.html
    ├── progress
    │   ├── Generation_0.csv
    │   └── Generation_1.csv
    └── results
        ├── logbook.csv
        └── populations.csv

Directory Contents
------------------

Each directory and file serves a specific purpose:

- **checkpoints**: Contains serialized checkpoint files for each generation. These checkpoints save the optimizer's state, allowing you to resume the process from a specific generation if necessary.
    - `cp_gen_0.pkl`, `cp_gen_1.pkl`: Checkpoints for each generation, named by generation number, saved in Python's pickle format.

- **graphics**: Stores HTML visualizations of the optimization process.
    - `logbook.html`: An interactive logbook visualization showing optimization statistics and trends over generations.
    - `search_space.html`: A visualization of the search space, showing how hyperparameters were explored during optimization.

- **progress**: Stores CSV files with detailed information about each generation's progress.
    - `Generation_0.csv`, `Generation_1.csv`: These files contain records of each individual in the population for each generation, including hyperparameters and fitness scores.

- **results**: Contains summary CSV files of the optimization results.
    - `logbook.csv`: A CSV version of the logbook, recording generation-by-generation statistics of the optimization process.
    - `populations.csv`: Final population data, including hyperparameters and fitness values of each individual in the last generation.

Each of these directories and files is structured to help you analyze and interpret the optimization process in detail, from individual generations to final results and visualizations.
