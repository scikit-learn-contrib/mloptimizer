"""
Linear Models Optimization
===========================
Hyperparameter optimization for Ridge, Lasso, and ElasticNet regression.
"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch
from mloptimizer.application.reporting.plots import plotly_search_space, plotly_logbook

# %%
# Load and prepare the dataset
print("Loading Diabetes dataset...")
data = load_diabetes()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# Optimize Ridge regression
print("\n=== Ridge Regression ===")
hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
    estimator_class=Ridge
)

opt = GeneticSearch(
    estimator_class=Ridge,
    hyperparam_space=hyperparam_space,
    generations=5,
    population_size=8,
    seed=42,
    use_mlflow=False,
    use_parallel=False
)

opt.fit(X_train, y_train)
y_pred = opt.best_estimator_.predict(X_test)
print(f"Ridge - Best params: {opt.best_params_}")
print(f"Ridge - Test R2: {r2_score(y_test, y_pred):.4f}")

# %%
# Visualize Ridge optimization
population_df = opt.populations_
g_logbook = plotly_logbook(opt.logbook_, population_df)
g_logbook.update_layout(
    title="Ridge Regression Optimization Evolution",
    autosize=True,
    width=None,
    height=500
)
plotly.io.show(g_logbook, config={'responsive': True})

# %%
# Optimize ElasticNet
print("\n=== ElasticNet Regression ===")
hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
    estimator_class=ElasticNet
)

opt = GeneticSearch(
    estimator_class=ElasticNet,
    hyperparam_space=hyperparam_space,
    generations=5,
    population_size=8,
    seed=42,
    use_mlflow=False,
    use_parallel=False
)

opt.fit(X_train, y_train)
y_pred = opt.best_estimator_.predict(X_test)
print(f"ElasticNet - Best params: {opt.best_params_}")
print(f"ElasticNet - Test R2: {r2_score(y_test, y_pred):.4f}")

# %%
# Visualize ElasticNet optimization
population_df = opt.populations_
g_logbook = plotly_logbook(opt.logbook_, population_df)
g_logbook.update_layout(
    title="ElasticNet Regression Optimization Evolution",
    autosize=True,
    width=None,
    height=500
)
plotly.io.show(g_logbook, config={'responsive': True})

# %%
# Analyze optimization performance
print("\n=== Optimization Performance ===")
print(f"Unique evaluations performed: {opt.n_trials_}")
print(f"Total individuals in population history: {len(opt.populations_)}")
print(f"Optimization time: {opt.optimization_time_:.4f} seconds")
print(f"Time per evaluation: {opt.optimization_time_ / opt.n_trials_:.4f} seconds")
