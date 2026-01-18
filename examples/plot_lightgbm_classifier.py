"""
LightGBM Classifier Optimization
=================================
Hyperparameter optimization for LightGBM using genetic algorithms on a classification task.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import plotly
from mloptimizer.interfaces import HyperparameterSpaceBuilder, GeneticSearch
from mloptimizer.application.reporting.plots import plotly_search_space, plotly_logbook

# Import LightGBM
try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("Please install lightgbm: pip install lightgbm")

# %%
# Load and prepare the dataset
# -----------------------------
print("Loading Breast Cancer dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
# Define the LightGBM hyperparameter space
# ----------------------------------------
# Use the default hyperparameter space for LGBMClassifier
hyperparam_space = HyperparameterSpaceBuilder.get_default_space(
    estimator_class=LGBMClassifier
)

# %%
# Configure and run the genetic optimization
# -------------------------------------------
# Note: Small values for fast execution. For production, increase generations and population_size.
genetic_params = {
    'generations': 5,
    'population_size': 8,
    'n_elites': 2,
    'seed': 42,
    'use_mlflow': False,
    'use_parallel': False
}

opt = GeneticSearch(
    estimator_class=LGBMClassifier,
    hyperparam_space=hyperparam_space,
    cv=3,
    scoring='accuracy',
    **genetic_params
)

print("Starting LightGBM optimization...")
opt.fit(X_train, y_train)

# %%
# Evaluate the optimized model
best_clf = opt.best_estimator_
y_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"\nOptimization completed!")
print(f"Best parameters: {opt.best_params_}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test F1: {f1:.4f}")

# %%
# Visualize the search space
population_df = opt.populations_
top_params = ['learning_rate', 'max_depth', 'n_estimators', 'num_leaves', 'fitness']
df_filtered = population_df[top_params]
g_search_space = plotly_search_space(df_filtered, top_params)
g_search_space.update_layout(
    title="LightGBM Hyperparameter Search Space",
    autosize=True,
    width=None,
    height=650
)
plotly.io.show(g_search_space, config={'responsive': True})

# %%
# Visualize the optimization evolution
g_logbook = plotly_logbook(opt.logbook_, population_df)
g_logbook.update_layout(
    title="LightGBM Optimization Evolution",
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

print(f"\nFitness Evolution:")
print(f"  Initial average fitness: {initial_avg_fitness:.4f}")
print(f"  Final average fitness: {final_avg_fitness:.4f}")
print(f"  Average fitness improvement: {improvement:.4f}")
