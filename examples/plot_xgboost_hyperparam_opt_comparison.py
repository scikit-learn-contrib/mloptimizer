"""
XGBoost - Genetic vs Grid Search vs Random Search vs Bayesian Optimization
==========================================================================
mloptimizer example optimization of iris dataset comparing hyperparameter tuning techniques:

1) Genetic optimization - mloptimizer
2) Grid Search - scikit-learn
3) Random Search - scikit-learn
4) Bayesian Optimization - hyperopt
"""

# %%
# Imports
# -------
# The necessary libraries for the example are imported.

import pandas as pd
import numpy as np
from time import time
import plotly

from mloptimizer.domain.hyperspace import HyperparameterSpace, Hyperparam
from mloptimizer.application.reporting.plots import plotly_search_space
from mloptimizer.interfaces import GeneticSearch

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, \
    StratifiedKFold
from sklearn.datasets import load_iris

from xgboost import XGBClassifier

from hyperopt import STATUS_OK, hp, tpe
from hyperopt import Trials, fmin

# %%
# 1) Dataset Description
# ----------------------
# The Iris dataset is a classic dataset used for classification tasks.
# It consists of 150 samples, each representing an iris flower.
#
# Features:
#
# - Sepal Length: Length of the sepal in centimeters.
# - Sepal Width: Width of the sepal in centimeters.
# - Petal Length: Length of the petal in centimeters.
# - Petal Width: Width of the petal in centimeters.
#
# Target:
#
# - Species of the flower, which can be one of the following three classes:
#
#   1. Setosa
#   2. Versicolor
#   3. Virginica
#
# Characteristics:
#
# - The dataset is balanced, containing 50 samples for each class.
# - Features are continuous and have varying scales, which may require normalization or standardization for certain machine learning algorithms.

name = 'iris'
X, y = load_iris(return_X_y=True)

print(f"1) Description of the dataset")
print(f"Dataset: {name}, X shape: {X.shape}, y shape: {y.shape}")

# %%
# 2) Genetic Search of XGBoost Algorithm
# ---------------------------------------
# Genetic search optimization is performed using the mloptimizer library to fine-tune the hyperparameters of the XGBoost
# algorithm.
#
# Hyperparameters to Optimize:
#
# - `colsample_bytree`: Subsample ratio of columns when constructing each tree.
# - `gamma`: Minimum loss reduction required to make a further partition on a leaf node of the tree.
# - `learning_rate`: Step size shrinkage used in updates to prevent overfitting.
# - `max_depth`: Maximum depth of a tree.
# - `n_estimators`: Number of boosting rounds.
# - `subsample`: Subsample ratio of the training instances.
#
# Optimization Process:
#
# - Population Size: 15
# - Generations: 10
# - Fitness Score: Balanced accuracy
# - Evaluation Function: Stratified k-fold cross-validation with 5 folds
#
# The genetic optimization explores the hyperparameter space defined by the evolvable hyperparameters.
# It searches within the defined minimum and maximum values for each hyperparameter.
#
# Advantages:
#
# - Provides a more exhaustive search compared to grid search and random search.
# - Can be more efficient in finding optimal hyperparameters for the XGBoost algorithm.
#
# Genetic Algorithm Configuration
# --------------------------------
# The following parameters control the behavior of the genetic algorithm:
#
# - `population_size`: Number of individuals (hyperparameter configurations) in each generation.
# - `generations`: Number of evolutionary iterations to perform.
# - `n_elites`: Number of best individuals to preserve unchanged in the next generation (elitism).
# - `tournsize`: Tournament size for selection (number of individuals competing in each tournament).
# - `cxpb`: Probability of mating (crossover) two individuals (0.0 to 1.0).
# - `mutpb`: Probability of mutating an individual (0.0 to 1.0).
# - `indpb`: Independent probability of mutating each hyperparameter within an individual (0.0 to 1.0).
# - `early_stopping`: Enable early stopping if fitness does not improve for a number of generations.
# - `patience`: Number of generations to wait for improvement before stopping early.
# - `min_delta`: Minimum change in fitness to be considered an improvement.
# - `seed`: Random seed for reproducibility.
# - `use_parallel`: Whether to use parallel evaluation of individuals.
#
# Note: Values reduced for faster documentation builds. For production comparison,
# use generations=20-30 and population_size=20-30 for more robust results.

print(f"2) Genetic Search optimization of XGBoost")

fixed_hyperparams = {}
evolvable_hyperparams = {
    'colsample_bytree': Hyperparam('colsample_bytree', 3, 10, 'float', 10),
    'gamma': Hyperparam('gamma', 0, 20, 'int'),
    'learning_rate': Hyperparam('learning_rate', 1, 100, 'float', 1000),
    'max_depth': Hyperparam('max_depth', 2, 20, 'int'),
    'n_estimators': Hyperparam('n_estimators', 100, 500, 'int'),
    'subsample': Hyperparam('subsample', 700, 1000, 'float', 1000)
}
hyperparameter_space = HyperparameterSpace(fixed_hyperparams, evolvable_hyperparams)

genetic_params = {
    'generations': 8,
    'population_size': 8,
    'n_elites': 2,
    'tournsize': 3,
    'cxpb': 0.5,
    'mutpb': 0.8,
    'indpb': 0.2,
    'early_stopping': True,
    'patience': 4,
    'min_delta': 0.005,
    'seed': 0,
    'use_parallel': False
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
opt = GeneticSearch(
    estimator_class=XGBClassifier,
    hyperparam_space=hyperparameter_space,
    cv=cv,
    scoring="balanced_accuracy",
    **genetic_params
)

t0_gen = time()
clf = opt.fit(X, y)
t1_gen = time()
execution_time_gen = round(t1_gen - t0_gen, 2)
print(f"Time of the Genetic Search optimization: {execution_time_gen} s")
population_df = opt.populations_
print(f"Genetic Search evaluated {population_df.shape[0]} configurations")
population_df_filtered = population_df[list(hyperparameter_space.evolvable_hyperparams.keys()) + ['fitness']]
fig_gen = plotly_search_space(population_df_filtered)
fig_gen.update_layout(autosize=True, width=None, height=650)
plotly.io.show(fig_gen, config={'responsive': True})

# %%
# 3) Grid Search Optimization for XGBoost
# ----------------------------------------
# Grid Search optimization is performed using the GridSearchCV class from the scikit-learn library.
#
# Optimization Process:
#
# - The hyperparameters to optimize are the same as those used in genetic optimization:
#
#   - `colsample_bytree`
#   - `gamma`
#   - `learning_rate`
#   - `max_depth`
#   - `n_estimators`
#   - `subsample`
#
# - Search Strategy:
#
#   - Grid Search performs an exhaustive search over a predefined hyperparameter space.
#   - The search space and number of parameter combinations are fixed by the user.
#
# Considerations:
#
# - Grid Search is computationally expensive, especially for large hyperparameter spaces.
# - It may not be feasible for extensive parameter tuning due to its exhaustive nature.

print(f"3) Grid Search optimization of XGBoost")

xgb = XGBClassifier()
# Reduced grid for faster documentation builds
parameters = {
    'colsample_bytree': (0.3, 0.6),
    'gamma': (5, 15),
    'learning_rate': (0.01, 0.1),
    'max_depth': (2, 8),
    'n_estimators': (200,),
    'subsample': (0.7, 0.9)
}
clf_gs = GridSearchCV(
    xgb,
    parameters,
    cv=cv,
    scoring="balanced_accuracy",
)

t0_gs = time()
clf_gs.fit(X, y)
t1_gs = time()
execution_time_gs = round(t1_gs - t0_gs, 2)
print(f"Time of the Grid Search optimization: {execution_time_gs} s")

synth_population_gs = pd.DataFrame(clf_gs.cv_results_['params'])
synth_population_gs['fitness'] = clf_gs.cv_results_['mean_test_score']
print(f"Grid Search evaluated {synth_population_gs.shape[0]} configurations")
fig_gs = plotly_search_space(synth_population_gs)
fig_gs.update_layout(autosize=True, width=None, height=650)
plotly.io.show(fig_gs, config={'responsive': True})

# %%
# 4) Random Search Optimization for XGBoost
# ------------------------------------------
# Random Search optimization is performed using the RandomizedSearchCV class from the scikit-learn library.
#
# Optimization Process:
#
# - The hyperparameters to optimize are the same as those used in genetic optimization and grid search:
#
#   - `colsample_bytree`
#   - `gamma`
#   - `learning_rate`
#   - `max_depth`
#   - `n_estimators`
#   - `subsample`
#
# - Search Strategy:
#
#   - Random Search samples a specified number of random combinations from the predefined hyperparameter space.
#   - The number of iterations (i.e., sampled parameter combinations) is fixed by the user.
#
# Considerations:
#
# - Random Search is less computationally expensive than Grid Search, making it more feasible for larger search spaces.
# - It is not as exhaustive as genetic optimization but can be more efficient in exploring the hyperparameter space compared to Grid Search.

print(f"4) Random Search optimization of XGBoost")

# Reduced search space for faster documentation builds
distributions = {
    'colsample_bytree': np.linspace(0.3, 1, 8),
    'gamma': (0, 5, 15),
    'learning_rate': (0.01, 0.1),
    'max_depth': (2, 5, 10),
    'n_estimators': (100, 200, 400),
    'subsample': np.linspace(0.7, 0.9, 8)
}
clf_rs = RandomizedSearchCV(
    xgb,
    distributions,
    cv=cv,
    n_iter=30,  # Reduced from 110 for faster builds
    random_state=0,
    scoring="balanced_accuracy"
)

t0_rs = time()
clf_rs.fit(X, y)
t1_rs = time()
execution_time_rs = round(t1_rs - t0_rs, 2)
print(f"Time of the Random Search optimization: {execution_time_rs} s")

synth_population_rs = pd.DataFrame(clf_rs.cv_results_['params'])
synth_population_rs['fitness'] = clf_rs.cv_results_['mean_test_score']
print(f"Random Search evaluated {synth_population_rs.shape[0]} configurations")
fig_rs = plotly_search_space(synth_population_rs)
fig_rs.update_layout(autosize=True, width=None, height=650)
plotly.io.show(fig_rs, config={'responsive': True})

# %%
# 5) Bayesian Optimization for XGBoost
# -------------------------------------
# Bayesian Optimization is performed using the hyperopt library to optimize the hyperparameters of the XGBoost algorithm.
#
# Optimization Process:
#
# - The hyperparameters to optimize are the same as those used in genetic optimization, grid search, and random search:
#
#   - `colsample_bytree`
#   - `gamma`
#   - `learning_rate`
#   - `max_depth`
#   - `n_estimators`
#   - `subsample`
#
# - Search Strategy:
#
#   - Bayesian Optimization uses a probabilistic model to predict the performance of different hyperparameter combinations.
#   - It iteratively evaluates the objective function (in this case, cross-validated balanced accuracy) to find the optimal hyperparameters.
#
# Considerations:
#
# - Bayesian Optimization is more efficient than random search and grid search for large search spaces.
# - It can provide better results with fewer evaluations compared to random search and grid search.
# - The number of evaluations is a parameter that can be adjusted to balance the trade-off between performance and computational cost.

print(f"5) Bayesian Optimization of XGBoost")

bayesian_space = {
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.3, 1.0),
    'gamma': hp.choice('gamma', np.arange(5, 50+1, dtype=int)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': hp.choice('max_depth', np.arange(2, 10+1, dtype=int)),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 500, dtype=int)),
    'subsample': hp.uniform('subsample', 0.6, 1.0)
}


def objective_function(params):
    clf = XGBClassifier(**params)
    score = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy").mean()
    return {'loss': -score, 'status': STATUS_OK}


trials = Trials()
num_eval = 30  # Reduced from 110 for faster documentation builds

t0_bay = time()
fmin(objective_function, bayesian_space, algo=tpe.suggest, max_evals=num_eval, trials=trials, verbose=0)
t1_bay = time()
execution_time_bay = round(t1_bay - t0_bay, 2)
print(f"Time of the Bayesian Optimization: {execution_time_bay} s")

# Extract parameters and scores from trials
trials_dict = trials.trials
params_list = [{k: v[0] for k, v in trial['misc']['vals'].items()} for trial in trials_dict]
scores_list = [-trial['result']['loss'] for trial in trials_dict]
synth_population_bay = pd.DataFrame(params_list)
synth_population_bay['fitness'] = scores_list
synth_population_bay.columns = [col.replace('vals_', '') for col in synth_population_bay.columns]
print(f"Bayesian Optimization evaluated {num_eval} configurations")
fig_bay = plotly_search_space(synth_population_bay)
fig_bay.update_layout(autosize=True, width=None, height=650)
plotly.io.show(fig_bay, config={'responsive': True})

# %%
# Summary Table
# -------------
# The summary table below compares the optimization methods based on their best metric,
# the time taken for optimization, and the number of evaluations performed.
#
# Table Columns:
#
# - **Method**: The optimization technique used (Genetic, Grid Search, Random Search, Bayesian).
# - **Best Metric**: The highest balanced accuracy score achieved by each method.
# - **Time**: The total execution time in seconds for the optimization process.
# - **Evaluated**: The number of hyperparameter configurations evaluated.
#
# This comparison allows you to assess the trade-offs between exploration efficiency,
# computational cost, and final model performance for each optimization strategy.

row_gen = {'Method': "Genetic",
           'Best Metric': float(population_df.sort_values(by="fitness", ascending=False)['fitness'].iloc[0]),
           'Time': execution_time_gen, 'Evaluated': population_df.shape[0]}
row_gs = {'Method': "Grid Search", 'Best Metric': clf_gs.best_score_,
          'Time': execution_time_gs, 'Evaluated': synth_population_gs.shape[0]}
row_rs = {'Method': "Random Search", 'Best Metric': clf_rs.best_score_,
          'Time': execution_time_rs, 'Evaluated': synth_population_rs.shape[0]}
row_bay = {'Method': "Bayesian Optimization",
           'Best Metric': max(scores_list),
           'Time': execution_time_bay, 'Evaluated': num_eval}

summary_df = pd.DataFrame([row_gen, row_gs, row_rs, row_bay])
print(summary_df)

# %%
# Best Hyperparameters Comparison
# --------------------------------
# This table shows the best hyperparameters found by each optimization method along with
# the achieved performance metric. Comparing these values helps understand:
#
# - **Performance vs Configuration**: How do hyperparameter choices relate to achieved scores?
# - **Convergence**: Do different methods find similar hyperparameter values for similar scores?
# - **Diversity**: Which hyperparameters vary the most across methods?
# - **Optimization Behavior**: How do different search strategies explore the space?
#
# Table Columns:
#
# - **Method**: The optimization technique used
# - **Best Metric**: The balanced accuracy score achieved with these hyperparameters
# - **colsample_bytree**: Subsample ratio of columns when constructing each tree
# - **gamma**: Minimum loss reduction required to make a further partition
# - **learning_rate**: Step size shrinkage used in updates
# - **max_depth**: Maximum depth of a tree
# - **n_estimators**: Number of boosting rounds
# - **subsample**: Subsample ratio of the training instances

# Extract best hyperparameters from each method
best_params_gen = opt.best_params_
best_params_gs = clf_gs.best_params_
best_params_rs = clf_rs.best_params_

# For Bayesian optimization, find the trial with the best score
best_trial_idx = np.argmax(scores_list)
best_params_bay = params_list[best_trial_idx]

# Extract best scores
best_score_gen = float(population_df.sort_values(by="fitness", ascending=False)['fitness'].iloc[0])
best_score_gs = clf_gs.best_score_
best_score_rs = clf_rs.best_score_
best_score_bay = max(scores_list)

# Create comparison dataframe
params_comparison = pd.DataFrame({
    'Method': ['Genetic', 'Grid Search', 'Random Search', 'Bayesian Optimization'],
    'Best Metric': [best_score_gen, best_score_gs, best_score_rs, best_score_bay],
    'colsample_bytree': [
        best_params_gen['colsample_bytree'],
        best_params_gs['colsample_bytree'],
        best_params_rs['colsample_bytree'],
        best_params_bay['colsample_by_tree']
    ],
    'gamma': [
        best_params_gen['gamma'],
        best_params_gs['gamma'],
        best_params_rs['gamma'],
        best_params_bay['gamma']
    ],
    'learning_rate': [
        best_params_gen['learning_rate'],
        best_params_gs['learning_rate'],
        best_params_rs['learning_rate'],
        best_params_bay['learning_rate']
    ],
    'max_depth': [
        best_params_gen['max_depth'],
        best_params_gs['max_depth'],
        best_params_rs['max_depth'],
        best_params_bay['max_depth']
    ],
    'n_estimators': [
        best_params_gen['n_estimators'],
        best_params_gs['n_estimators'],
        best_params_rs['n_estimators'],
        best_params_bay['n_estimators']
    ],
    'subsample': [
        best_params_gen['subsample'],
        best_params_gs['subsample'],
        best_params_rs['subsample'],
        best_params_bay['subsample']
    ]
})

print("\nBest Hyperparameters Found by Each Method:")
print(params_comparison.to_string(index=False))

# %%
# Combined Summary
# ----------------
# This final table combines both the performance metrics and the best hyperparameters
# found by each optimization method for a comprehensive comparison.

combined_summary = summary_df.copy()
for param in ['colsample_bytree', 'gamma', 'learning_rate', 'max_depth', 'n_estimators', 'subsample']:
    combined_summary[param] = params_comparison[param]

combined_summary
