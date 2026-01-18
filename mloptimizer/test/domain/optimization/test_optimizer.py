import pytest
import warnings
from mloptimizer.domain.optimization import Optimizer
from mloptimizer.domain.hyperspace import Hyperparam, HyperparameterSpace
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from xgboost import XGBClassifier
from sklearn.svm import SVC
from mloptimizer.domain.evaluation import kfold_score, train_score, train_test_score
import time
import os
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

custom_evolvable_hyperparams = {
    "min_samples_split": Hyperparam("min_samples_split", 2, 50, 'int'),
    "min_samples_leaf": Hyperparam("min_samples_leaf", 1, 20, 'int'),
    "max_depth": Hyperparam("max_depth", 2, 20, 'int'),
    "min_impurity_decrease": Hyperparam("min_impurity_decrease", 0,
                                        150, 'float', 1000),
    "ccp_alpha": Hyperparam("ccp_alpha", 0, 300, 'float', 100000)
}


@pytest.fixture
def default_metrics_dict():
    return {
        "accuracy": accuracy_score,
    }


@pytest.mark.parametrize('estimator_class',
                         (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                          GradientBoostingClassifier, XGBClassifier, SVC))
def test_sklearn_optimizer(estimator_class, tmp_path):
    x, y = load_iris(return_X_y=True)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    mlopt = Optimizer(estimator_class=estimator_class,
                      hyperparam_space=evolvable_hyperparams,
                      features=x, labels=y, folder=tmp_path)
    mlopt.optimize_clf(5, 5)
    assert mlopt is not None


@pytest.mark.parametrize('use_mlflow', [True, False])
def test_mloptimizer(use_mlflow, tmp_path):
    x, y = load_breast_cancer(return_X_y=True)
    mlopt = Optimizer(estimator_class=XGBClassifier,
                      hyperparam_space=HyperparameterSpace.get_default_hyperparameter_space(XGBClassifier),
                      genetic_params={"population_size": 5, "generations": 5},
                      features=x, labels=y, use_mlflow=use_mlflow, folder=tmp_path)
    mlopt.optimize_clf(5, 5)
    assert mlopt is not None


def test_checkpoints():
    x, y = load_breast_cancer(return_X_y=True)
    mlopt = Optimizer(estimator_class=XGBClassifier,
                      hyperparam_space=HyperparameterSpace.get_default_hyperparameter_space(XGBClassifier),
                      genetic_params={"population_size": 5, "generations": 5},
                      features=x, labels=y, disable_file_output=False)
    clf = mlopt.optimize_clf(5, 5)

    mlopt2 = Optimizer(estimator_class=XGBClassifier,
                       hyperparam_space=HyperparameterSpace.get_default_hyperparameter_space(XGBClassifier),
                       genetic_params={"population_size": 5, "generations": 5},
                       features=x, labels=y,
                       seed=mlopt.mlopt_seed)

    checkpoint = os.path.join(mlopt.tracker.opt_run_checkpoint_path,
                              os.listdir(mlopt.tracker.opt_run_checkpoint_path)[-2]
                              )
    clf2 = mlopt2.optimize_clf(5, 5,
                               checkpoint=checkpoint)
    assert mlopt is not None
    assert str(clf) == str(clf2)


@pytest.mark.parametrize('estimator_class',
                         (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                          GradientBoostingClassifier, XGBClassifier, SVC))
@pytest.mark.parametrize('dataset',
                         (load_breast_cancer, load_iris))
def test_optimizer(estimator_class, dataset, default_metrics_dict, tmp_path):
    x, y = dataset(return_X_y=True)
    if estimator_class == SVC:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    opt = Optimizer(features=x, labels=y, fitness_score="accuracy",
                    metrics=default_metrics_dict, estimator_class=estimator_class,
                    genetic_params={"generations": 2, "population_size": 2}, seed=42,
                    hyperparam_space=evolvable_hyperparams, folder=tmp_path)
    clf = opt.optimize_clf(2, 2)
    assert clf is not None


@pytest.mark.parametrize('estimator_class',
                         (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                          GradientBoostingClassifier, SVC))
@pytest.mark.parametrize('dataset',
                         (load_breast_cancer, load_iris))
def test_optimizer_use_parallel(estimator_class, dataset, default_metrics_dict, tmp_path):
    x, y = dataset(return_X_y=True)
    if estimator_class == SVC:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    my_seed = 25
    population = 60
    generations = 4

    opt_with_parallel = Optimizer(features=x, labels=y, fitness_score="accuracy",
                                  metrics=default_metrics_dict,
                                  seed=my_seed, use_parallel=True,
                                  hyperparam_space=evolvable_hyperparams, estimator_class=estimator_class,
                                    genetic_params={"population_size": population, "generations": generations},
                                  folder=tmp_path)

    start_time_parallel = time.time()
    clf_with_parallel = opt_with_parallel.optimize_clf(population, generations)
    end_time_parallel = time.time()

    opt = Optimizer(features=x, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                    seed=my_seed, use_parallel=False,
                    hyperparam_space=evolvable_hyperparams, estimator_class=estimator_class,
                    genetic_params={"population_size": population, "generations": generations},
                    folder=tmp_path)

    start_time = time.time()
    clf = opt.optimize_clf(population, generations)
    end_time = time.time()

    elapsed_time_parallel = end_time_parallel - start_time_parallel
    elapsed_time = end_time - start_time
    speedup = round(((elapsed_time_parallel / elapsed_time) - 1) * 100, 2)

    print(f"Elapsed time with parallel: {elapsed_time_parallel}")
    print(f"Elapsed time without parallel: {elapsed_time}")
    print(f"Speedup: {speedup}%")
    assert str(clf) == str(clf_with_parallel)
    if elapsed_time_parallel < elapsed_time:
        warnings.warn(
            f"Sequential execution time ({elapsed_time:.2f}s) is greater than or equal "
            f"to parallel execution time ({elapsed_time_parallel:.2f}s)."
        )


@pytest.mark.parametrize('estimator_class',
                         (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                          GradientBoostingClassifier, XGBClassifier, SVC))
@pytest.mark.parametrize('target_score', (kfold_score, train_score, train_test_score))
def test_reproducibility(estimator_class, target_score, default_metrics_dict, tmp_path):
    x, y = load_iris(return_X_y=True)
    if estimator_class == SVC:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    evolvable_hyperparams = HyperparameterSpace.get_default_hyperparameter_space(estimator_class)
    population = 2
    generations = 2
    seed = 25
    distinct_seed = 2
    optimizer1 = Optimizer(features=x, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                           eval_function=target_score, seed=seed, estimator_class=estimator_class,
                           genetic_params={"generations": generations, "population_size": population},
                           hyperparam_space=evolvable_hyperparams, folder=tmp_path)
    result1 = optimizer1.optimize_clf(population_size=population, generations=generations)
    optimizer2 = Optimizer(features=x, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                           eval_function=target_score, seed=seed, estimator_class=estimator_class,
                           genetic_params={"generations": generations, "population_size": population},
                           hyperparam_space=evolvable_hyperparams, folder=tmp_path)
    result2 = optimizer2.optimize_clf(population_size=population, generations=generations)
    optimizer3 = Optimizer(features=x, labels=y, fitness_score="accuracy", metrics=default_metrics_dict,
                           eval_function=target_score, seed=distinct_seed, estimator_class=estimator_class,
                           genetic_params={"generations": generations, "population_size": population},
                           hyperparam_space=evolvable_hyperparams, folder=tmp_path)
    result3 = optimizer3.optimize_clf(population_size=population, generations=generations)
    assert str(result1) == str(result2)
    assert str(result1) != str(result3)


def test_custom_svc():
    x, y = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    custom_svc_evolvable_hyperparams = {
        "C": Hyperparam("C", -2, 2, 'nexp'),
        "kernel": Hyperparam("kernel", 0, 3, 'list',
                             values_str=['linear', 'poly', 'rbf', 'sigmoid']),
        "degree": Hyperparam("degree", 0, 6, 'int'),
        "gamma": Hyperparam("gamma", -3, 3, 'nexp'),
        "max_iter": Hyperparam("max_iter", 1000, 10000, 'int')
    }
    evolvable_hyperparams = HyperparameterSpace(evolvable_hyperparams=custom_svc_evolvable_hyperparams,
                                                fixed_hyperparams={}
                                                )
    opt = Optimizer(features=x, labels=y, fitness_score="balanced_accuracy",
                    metrics={"balanced_accuracy": accuracy_score},
                    estimator_class=SVC,
                    hyperparam_space=evolvable_hyperparams,
                    genetic_params={"generations": 4, "population_size": 60}, seed=42,
                    use_parallel=False)
    clf = opt.optimize_clf(60, 4)
    assert clf is not None

def test_validate_hyperparam_space_none():
    x, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="hyperparam_space is None"):
        Optimizer(estimator_class=DecisionTreeClassifier, hyperparam_space=None,
                  genetic_params={},
                  features=x, labels=y)

def test_validate_hyperparam_space_invalid_params():
    x, y = load_iris(return_X_y=True)
    invalid_hyperparams = {
        "invalid_param": Hyperparam("invalid_param", 0, 10, 'int')
    }
    invalid_fixed_hyperparams = {
        "max_depth": 10
    }
    hyperparam_space = HyperparameterSpace(evolvable_hyperparams=invalid_hyperparams,
                                           fixed_hyperparams=invalid_fixed_hyperparams)
    with pytest.raises(ValueError,
                       match="Parameters {'invalid_param'} are not parameters of DecisionTreeClassifier"):
        Optimizer(estimator_class=DecisionTreeClassifier, hyperparam_space=hyperparam_space,
                  genetic_params={}, features=x, labels=y)

def test_validate_hyperparam_space_valid_params():
    x, y = load_iris(return_X_y=True)
    valid_hyperparams = {
        "max_depth": Hyperparam("max_depth", 1, 10, 'int')
    }
    hyperparam_space = HyperparameterSpace(evolvable_hyperparams=valid_hyperparams,
                                           fixed_hyperparams={})
    optimizer = Optimizer(estimator_class=DecisionTreeClassifier, hyperparam_space=hyperparam_space,
                          genetic_params={}, features=x, labels=y)
    assert optimizer is not None