"""
Search space graph
====================================
mloptimizer provides a function to plot the search space of the optimization.
"""

from sklearn.tree import DecisionTreeClassifier
from mloptimizer.application.reporting.plots import plotly_search_space
import plotly
import os
from sklearn.datasets import load_iris
from mloptimizer.interfaces import GeneticSearch
from mloptimizer.domain.hyperspace import HyperparameterSpace, Hyperparam
from sklearn.model_selection import StratifiedKFold


# %%
# Load the iris dataset to obtain a vector of features X and a vector of labels y.
# Another dataset or a custom one can be used
X, y = load_iris(return_X_y=True)

# %%
# Define the HyperparameterSpace using the dictionary approach
# -------------------------------------------------------------
# This demonstrates the direct way to define hyperparameters as the library specifies.
# Each Hyperparam defines: name, min_value, max_value, type, and optional scale.
fixed_hyperparams = {}
evolvable_hyperparams = {
    'max_depth': Hyperparam('max_depth', 1, 20, 'int'),
    'min_samples_split': Hyperparam('min_samples_split', 2, 20, 'int'),
    'min_samples_leaf': Hyperparam('min_samples_leaf', 1, 20, 'int'),
    'max_features': Hyperparam('max_features', 10, 100, 'float', 100)
}
hyperparam_space = HyperparameterSpace(fixed_hyperparams, evolvable_hyperparams)

# %%
# The GeneticSearch class is used to optimize the hyperparameters of a machine learning model.
# Configure genetic algorithm parameters for reproducible results.
# Note: Values reduced for faster documentation builds. For production, use larger values.
genetic_params = {
    'generations': 10,
    'population_size': 20,
    'n_elites': 2,
    'cxpb': 0.5,
    'mutpb': 0.8,
    'seed': 42
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
opt = GeneticSearch(
    estimator_class=DecisionTreeClassifier,
    hyperparam_space=hyperparam_space,
    cv=cv,
    scoring='accuracy',
    **genetic_params
)

# %%
# To optimize the classifier we need to call the fit method.
opt.fit(X, y)
clf = opt.best_estimator_


# %%
# Following we can generate the plot of the search space
population_df = opt.populations_
param_names = list(opt.get_evolvable_hyperparams().keys())
param_names.append("fitness")
df = population_df[param_names]
g_search_space = plotly_search_space(df, param_names)
g_search_space.update_layout(autosize=True, width=None, height=600)
plotly.io.show(g_search_space, config={'responsive': True})

# %%
# At the end of the evolution the graph is saved as an html at the path:
print(opt._optimizer_service.optimizer.tracker.graphics_path)
print(os.listdir(opt._optimizer_service.optimizer.tracker.graphics_path))


# %%
# The data to generate the graph is available at the path:
print(opt._optimizer_service.optimizer.tracker.results_path)
print(os.listdir(opt._optimizer_service.optimizer.tracker.results_path))

del opt
