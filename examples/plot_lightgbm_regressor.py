"""
LightGBM Regressor Optimization
================================
Hyperparameter optimization for LightGBM regressor using genetic algorithms.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch
from mloptimizer.application.reporting.plots import plotly_search_space, plotly_logbook

# Import LightGBM
try:
    from lightgbm import LGBMRegressor
except ImportError:
    raise ImportError("Please install lightgbm: pip install lightgbm")

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
# Define the LightGBM hyperparameter space
hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
    estimator_class=LGBMRegressor
)

# %%
# Configure and run the genetic optimization
genetic_params = {
    'generations': 5,
    'population_size': 8,
    'n_elites': 2,
    'seed': 42,
    'use_mlflow': False,
    'use_parallel': False
}

opt = GeneticSearch(
    estimator_class=LGBMRegressor,
    hyperparam_space=hyperparam_space,
    cv=3,
    **genetic_params
)

print("Starting LightGBM Regressor optimization...")
opt.fit(X_train, y_train)

# %%
# Evaluate the optimized model
best_reg = opt.best_estimator_
y_pred = best_reg.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"\nOptimization completed!")
print(f"Best parameters: {opt.best_params_}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test R2: {test_r2:.4f}")

# %%
# Visualize the search space
population_df = opt.populations_
top_params = ['learning_rate', 'max_depth', 'n_estimators', 'num_leaves', 'fitness']
df_filtered = population_df[top_params]
g_search_space = plotly_search_space(df_filtered, top_params)
g_search_space.update_layout(
    title="LightGBM Regressor Hyperparameter Search Space",
    autosize=True,
    width=None,
    height=650
)
plotly.io.show(g_search_space, config={'responsive': True})

# %%
# Visualize the optimization evolution
g_logbook = plotly_logbook(opt.logbook_, population_df)
g_logbook.update_layout(
    title="LightGBM Regressor Optimization Evolution",
    autosize=True,
    width=None,
    height=500
)
plotly.io.show(g_logbook, config={'responsive': True})

# %%
# Analyze optimization results
print("\n=== Optimization Performance ===")
print(f"Unique evaluations performed: {opt.n_trials_}")
print(f"Total individuals in population history: {len(population_df)}")
print(f"Optimization time: {opt.optimization_time_:.4f} seconds")
print(f"Time per evaluation: {opt.optimization_time_ / opt.n_trials_:.4f} seconds")
