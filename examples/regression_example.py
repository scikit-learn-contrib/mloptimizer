"""
Regression Example
==================
This example shows how to optimize a regression model using the mloptimizer library.
"""

# %%
# Load the diabetes dataset
# -------------------------
# The diabetes dataset is used to demonstrate the optimization of a regression model.
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

# %%
# The dataset has 10 features, but we will use only one for this plot (the Body Mass Index (BMI)).
import matplotlib.pyplot as plt
plt.scatter(X[:,2], y)
plt.xlabel("Body Mass Index")
plt.ylabel("Disease Progression")
plt.title("Diabetes Dataset")
plt.show()

# %%
# Default hyperparameter space
# ----------------------------
# Define the HyperparameterSpace, you can use the default hyperparameters for the machine learning model
# that you want to optimize. In this case we use the default hyperparameters for a RandomForestRegressor.
from mloptimizer.domain.hyperspace import HyperparameterSpace
from sklearn.ensemble import RandomForestRegressor

hyperparam_space = HyperparameterSpace.get_default_hyperparameter_space(RandomForestRegressor)

# %%
# We use the GeneticSearch class to optimize a regression model.
# Configure the genetic algorithm with default parameters shown explicitly.
# Note: Values reduced for faster documentation builds. For production, use larger values.
from mloptimizer.interfaces import GeneticSearch

genetic_params = {
    'generations': 8,
    'population_size': 15,
    'n_elites': 2,
    'seed': 42
}

mlopt = GeneticSearch(
    estimator_class=RandomForestRegressor,
    hyperparam_space=hyperparam_space,
    cv=5,
    scoring='rmse',  # Root Mean Squared Error (available: 'rmse', 'mse', 'mae')
    **genetic_params
)

mlopt.fit(X, y)
clf = mlopt.best_estimator_
# %%
# The best estimator with optimized hyperparameters is stored in ``best_estimator_``.
# This estimator is trained with the best hyperparameters found during optimization.
# As the literature suggests, you can retrain with all the data or use cross-validation
# estimators for making predictions with better generalization.
print(clf)

# %%
# Custom hyperparameter space
# ---------------------------
# The hyperparameter space can be defined by the user.
# In this example, we define a new hyperparameter space for the RandomForestRegressor model.
from mloptimizer.domain.hyperspace import Hyperparam
fixed_hyperparams = {'n_estimators': 100}
evolvable_hyperparams = {'max_depth': Hyperparam(name='max_depth', min_value=1,
                                                 max_value=10, hyperparam_type='int'),
                         'min_samples_split': Hyperparam(name='min_samples_split', min_value=2,
                                                         max_value=10, hyperparam_type='int'),
                         'min_samples_leaf': Hyperparam(name='min_samples_leaf', min_value=1,
                                                        max_value=10, hyperparam_type='int')}
custom_hyperparam_space = HyperparameterSpace(fixed_hyperparams, evolvable_hyperparams)


# %%
# Furthermore, it is possible to set the metrics used to evaluate the model.
# We configure the genetic algorithm for this custom hyperparameter space.
# Note: Values reduced for faster documentation builds.
genetic_params_custom = {
    'generations': 8,
    'population_size': 15,
    'n_elites': 2,
    'cxpb': 0.5,
    'mutpb': 0.8,
    'seed': 42
}

mlopt = GeneticSearch(
    estimator_class=RandomForestRegressor,
    hyperparam_space=custom_hyperparam_space,
    cv=5,
    scoring='rmse',  # Root Mean Squared Error
    **genetic_params_custom
)

mlopt.fit(X, y)

print(mlopt.best_estimator_)