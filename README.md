![mloptimizer_banner](https://raw.githubusercontent.com/Caparrini/mloptimizer-static/main/logos/mloptimizer_banner_readme.png)

[![Documentation Status](https://readthedocs.org/projects/mloptimizer/badge/?version=master)](https://mloptimizer.readthedocs.io/en/master/?badge=master)
[![PyPI version](https://badge.fury.io/py/mloptimizer.svg)](https://badge.fury.io/py/mloptimizer)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mloptimizer.svg)](https://pypi.python.org/pypi/mloptimizer/)
[![Tests](https://github.com/Caparrini/mloptimizer/actions/workflows/CI.yml/badge.svg)](https://github.com/Caparrini/mloptimizer/actions/workflows/CI.yml)
[![Coverage Status](http://codecov.io/github/Caparrini/mloptimizer/coverage.svg?branch=master)](https://app.codecov.io/gh/Caparrini/mloptimizer)
[![Shield: Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-Support-yellow?logo=buymeacoffee)](https://www.buymeacoffee.com/caparrini)



**mloptimizer** is a Python library for optimizing hyperparameters of machine learning algorithms using genetic algorithms. 
With mloptimizer, you can find the optimal set of hyperparameters for a given machine learning model and dataset, which can significantly improve the performance of the model. 
The library supports several popular machine learning algorithms, including decision trees, random forests, and gradient boosting classifiers. 
The genetic algorithm used in mloptimizer provides an efficient and flexible approach to search for the optimal hyperparameters in a large search space.

## Features
- Easy to use
- DEAP-based genetic algorithm ready to use with several machine learning algorithms
- Adaptable to use with any machine learning algorithm that complies with the Scikit-Learn API
- Default hyperparameter ranges
- Default score functions for evaluating the performance of the model
- Reproducibility of results
- Early stopping to prevent overfitting
- Population seeding with known good configurations
- Performance tracking (trials count, optimization time)
- Zero file output mode for cleaner workflows (enabled by default)

## Advanced Features
- Extensible with more machine learning algorithms that comply with the Scikit-Learn API
- Customizable hyperparameter ranges
- Customizable score functions
- Optional mlflow compatibility for tracking the optimization process
- Generation-level MLflow tracking with nested runs
- Responsive Plotly visualizations with WebGL acceleration
- Joblib-based parallelization for better compatibility

## Installation

It is recommended to create a virtual environment using the `venv` package. 
To learn more about how to use `venv`, 
check out the official Python documentation at 
https://docs.python.org/3/library/venv.html.

```bash
# Create the virtual environment
python -m venv myenv
# Activate the virtual environment
source myenv/bin/activate
```

To install `mloptimizer`, run:

```bash
pip install mloptimizer
```

You can get more information about the package installation at https://pypi.org/project/mloptimizer/.


### Quickstart

Here's a simple example of how to optimize hyperparameters in a decision tree classifier using the iris dataset:

```python
from mloptimizer.interfaces import GeneticSearch, HyperparameterSpaceBuilder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# 1) Load the dataset and get the features and target
X, y = load_iris(return_X_y=True)

# 2) Define the hyperparameter space (a default space is provided for some algorithms)
hyperparameter_space = HyperparameterSpaceBuilder.get_default_space(DecisionTreeClassifier)

# 3) Create the optimizer and optimize the classifier
opt = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparameter_space,
    generations=10,
    population_size=20,
    early_stopping=True,        # Stop early if no improvement
    patience=3,                  # Wait 3 generations
    cv=5,                        # 5-fold cross-validation
    seed=42                      # Reproducibility
)

# 4) Optimize (no files created by default)
opt.fit(X, y)

# Access results
print(f"Best score: {opt.best_estimator_.score(X, y)}")
print(f"Trials evaluated: {opt.n_trials_}")
print(f"Time taken: {opt.optimization_time_:.2f}s")
```
Other algorithms can be used, such as `RandomForestClassifier` or `XGBClassifier` which have a 
default hyperparameter space defined in the library.
Even if the algorithm is not included in the default hyperparameter space, you can define your own hyperparameter space
following the documentation.


More details in the [documentation](http://mloptimizer.readthedocs.io/).

## Examples

Examples can be found in [examples](https://mloptimizer.readthedocs.io/en/master/auto_examples/index.html) on readthedocs.io.

## Dependencies

The following dependencies are used in `mloptimizer`:

* [Deap](https://github.com/DEAP/deap) - Genetic Algorithms
* [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting framework
* [CatBoost](https://catboost.ai/) - Gradient boosting framework
* [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting framework
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) - Machine learning algorithms and utilities
* [Plotly](https://plotly.com/python/) - Interactive visualizations
* [Seaborn](https://seaborn.pydata.org/) - Statistical visualizations
* [joblib](https://joblib.readthedocs.io/) - Parallel processing
* [tqdm](https://github.com/tqdm/tqdm) - Progress bars

Optional:
* [Keras](https://keras.io) - Deep learning library
* [mlflow](https://mlflow.org) - Tracking the optimization process

## Documentation

The documentation for `mloptimizer` can be found in the project's [wiki](http://mloptimizer.readthedocs.io/)
with examples, classes and methods reference.


## Authors

* **Antonio Caparrini** - *Author* - [caparrini](https://github.com/caparrini)
* **Javier Arroyo Gallardo** - *Author* - [javiag](https://github.com/javiag)

## Analytics

![Alt](https://repobeats.axiom.co/api/embed/e971cafaa4d71a2f24df2ede80714b2e2d06901f.svg "Repobeats analytics image")

## License

This project is licensed under the [MIT License](LICENSE).
