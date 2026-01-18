"""
XGBoost optimization with MLflow tracking
=========================================
A complete example showing hyperparameter optimization for XGBoost with MLflow integration
for experiment tracking.
"""

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import numpy as np
import plotly
import os
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch
from mloptimizer.application.reporting.plots import (plotly_search_space, plotly_logbook,
                                                     plot_logbook)
import matplotlib.pyplot as plt

# %%
# Load and prepare a complex classification dataset
# -------------------------------------------------
print("Loading Forest CoverType dataset...")
data = fetch_covtype()
X, y = data.data, data.target
y = y - 1  # Adjust labels to start from 0
# Use a subset for faster execution
np.random.seed(42)
sample_indices = np.random.choice(len(X), size=2000, replace=False)
X = X[sample_indices]
y = y[sample_indices]

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
# Define the XGBoost hyperparameter space using HyperparameterSpaceBuilder
# -------------------------------------------------------------------------
# We can build a custom hyperparameter space by adding individual parameters.
# This gives fine-grained control over the search space for each hyperparameter.
hyperparam_space = (HyperparameterSpaceBuilder()
                    .add_int_param('max_depth', min_value=2, max_value=10)
                    .add_float_param('learning_rate', min_value=10, max_value=30, scale=100)
                    .add_int_param('n_estimators', min_value=50, max_value=300)
                    .add_float_param('subsample', min_value=60, max_value=100, scale=100)
                    .add_float_param('colsample_bytree', min_value=60, max_value=100, scale=100)
                    .build())

# %%
# Configure and run the genetic optimization WITH MLFLOW
# ------------------------------------------------------
# Genetic Algorithm Configuration:
# - generations: Number of evolutionary iterations
# - population_size: Number of configurations per generation
# - n_elites: Number of best individuals preserved each generation
# - seed: Random seed for reproducibility
# - use_mlflow: Enable MLflow experiment tracking
# Note: Small values for documentation builds. For production, increase to 20+ generations.
genetic_params = {
    'generations': 5,
    'population_size': 8,
    'n_elites': 2,
    'seed': 42,
    'use_mlflow': True,
    'use_parallel': False
}

opt = GeneticSearch(
    estimator_class=xgb.XGBClassifier,
    hyperparam_space=hyperparam_space,
    cv=3,
    scoring='accuracy',
    disable_file_output=False,
    **genetic_params
)

print("Starting XGBoost optimization with MLflow tracking...")
print(f"use_mlflow parameter: {opt.use_mlflow}")

# Run the optimization
opt.fit(X_train, y_train)

# %%
# Get results and evaluate
best_clf = opt.best_estimator_
y_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nOptimization completed!")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test F1: {f1:.4f}")

# %%
# Generate visualizations
population_df = opt.populations_

# Search space visualization
top_params = ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'fitness']
df_filtered = population_df[top_params]
g_search_space = plotly_search_space(df_filtered, top_params)
g_search_space.update_layout(
    title="XGBoost Hyperparameter Search Space - CoverType Dataset",
    autosize=True,
    width=None,
    height=650
)
plotly.io.show(g_search_space, config={'responsive': True})

# %%
# Simple logbook visualization
g_logbook_s = plot_logbook(opt.logbook_)
# Show plot
plt.show()

# %%
# Evolution logbook visualization
g_logbook = plotly_logbook(opt.logbook_, population_df)
g_logbook.update_layout(
    title="XGBoost Optimization Evolution - CoverType Dataset",
    autosize=True,
    width=None,
    height=500
)
plotly.io.show(g_logbook, config={'responsive': True})

# %%
# Analyze optimization results
print("\n=== Optimization Analysis ===")
print(f"Unique evaluations performed: {opt.n_trials_}")
print(f"Total individuals in population history: {len(population_df)}")
print(f"Optimization time: {opt.optimization_time_:.4f} seconds")
print(f"Time per evaluation: {opt.optimization_time_ / opt.n_trials_:.4f} seconds")
print(f"Generations completed: {opt.generations}")

final_gen = population_df[population_df['population'] == opt.generations]
initial_gen = population_df[population_df['population'] == 1]

final_avg_fitness = final_gen['fitness'].mean()
initial_avg_fitness = initial_gen['fitness'].mean()
improvement = final_avg_fitness - initial_avg_fitness

print(f"Average fitness improvement: {improvement:.4f}")
print(f"Initial average fitness: {initial_avg_fitness:.4f}")
print(f"Final average fitness: {final_avg_fitness:.4f}")

# %%
 # Access generated files
print("\n=== Generated Files ===")
graphics_path = opt._optimizer_service.optimizer.tracker.graphics_path
results_path = opt._optimizer_service.optimizer.tracker.results_path

print(f"Graphics path: {graphics_path}")
if os.path.exists(graphics_path):
    print("Graphics files:", [f for f in os.listdir(graphics_path) if f.endswith('.html')])

print(f"Results path: {results_path}")
if os.path.exists(results_path):
    print("Results files:", [f for f in os.listdir(results_path) if f.endswith('.csv')])

# %%
# MLflow UI Instructions
# ----------------------
#
# To inspect the results recorded during the optimization, you can launch the
# MLflow user interface from a terminal.
#
# **Starting the MLflow UI**
#
# Open a console and run::
#
#   mlflow ui --port 5000
#
# Then open a web browser and go to:
#
#   http://localhost:5000
#
# **In the MLflow UI you can**
#
# - View all optimization runs in the experiment
# - Compare hyperparameters and metrics across runs
# - See the evolution of fitness scores across generations
# - Inspect logs and stored artifacts (TODO)
# - Track model performance and optimization progress