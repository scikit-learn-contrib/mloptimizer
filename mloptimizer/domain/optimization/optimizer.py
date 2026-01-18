import os
import random
import inspect
import numpy as np
from sklearn.base import is_regressor, is_classifier

from mloptimizer.domain.evaluation import train_score
from mloptimizer.domain.hyperspace import HyperparameterSpace
from mloptimizer.infrastructure.tracking import Tracker
from mloptimizer.domain.evaluation import Evaluator
from mloptimizer.__main__ import get_version



class Optimizer:
    """
    Base class for the optimization of a classifier

    Attributes
    ----------
    estimator_class : class
        class of the classifier
    features : np.array
        np.array with the features
    labels : np.array
        np.array with the labels
    hyperparam_space : HyperparameterSpace
        object with the hyperparameter space: fixed and evolvable hyperparams
    genetic_params : dict
        dictionary with the parameters of the genetic algorithm
    evaluator : Evaluator
        object to evaluate the classifier
    eval_dict : dict
        dictionary with the evaluation of the individuals
    populations : list
        list of populations
    logbook : list
        list of logbook
    seed : int
        seed for the random functions
    use_parallel : bool
        flag to use parallel processing
    use_mlflow : bool
        flag to use mlflow
    early_stopping : bool, optional (default=False)
        If True, the optimization will stop early if no improvement is observed in the fitness score.
    patience : int, optional (default=5)
        Number of generations to wait before stopping if no improvement is observed.
    min_delta : float, optional (default=0.01)
        Minimum change in the fitness score to qualify as an improvement.
    """

    def __init__(self, estimator_class, features: np.array, labels: np.array,
                 folder=os.curdir,
                 hyperparam_space: HyperparameterSpace = None,
                 genetic_params: dict = None,
                 eval_function: callable = train_score,
                 fitness_score=None, metrics=None, seed=random.randint(0, 1000000),
                 use_parallel=False, use_mlflow=False, disable_file_output=True,
                 early_stopping: bool = False, patience: int = 5, min_delta: float = 0.01,
                 initial_params: list = None, include_default: bool = False):
        """
        Creates object BaseOptimizer.

        Parameters
        ----------
        estimator_class : class
            class of the classifier
        features : np.array
            np.array with the features
        labels : np.array
            np.array with the labels
        folder : path, optional (default=os.curdir)
            folder to store the structure of files and folders product of executions
        hyperparam_space : HyperparameterSpace, optional (default=None)
            object with the hyperparameter space: fixed and evolvable hyperparams
        genetics_params : dict, optional (default=None)
            dictionary with the parameters of the genetic algorithm
        eval_function : func, optional (default=train_score)
            function to evaluate the model from X, y, clf
        fitness_score : str, optional (default="accuracy")
            fitness score to use to evaluate the performance of the classifier
        use_parallel : bool, optional (default=False)
            flag to use parallel processing
        use_mlflow : bool, optional (default=False)
            flag to use mlflow
        seed : int, optional (default=0)
            seed for the random functions (deap, models, and splits on evaluations)
        early_stopping : bool, optional (default=False)
            If True, the optimization will stop early if no improvement is observed in the fitness score.
        patience : int, optional (default=5)
            Number of generations to wait before stopping if no improvement is observed.
        min_delta : float, optional (default=0.01)
            Minimum change in the fitness score to qualify as an improvement.
        initial_params : list of dict, optional (default=None)
            List of hyperparameter dictionaries to seed the initial population with.
        include_default : bool, optional (default=False)
            If True, include an individual representing sklearn defaults in the initial population.
        """
        from mloptimizer.domain.population import IndividualUtils
        # Model class
        self.estimator_class = estimator_class
        if not is_classifier(self.estimator_class) and not is_regressor(self.estimator_class):
            raise ValueError(f"The estimator class {self.estimator_class} must be a classifier or a regressor")

        # Input mandatory variables
        self.features = features
        self.labels = labels
        # Input search space hyperparameters
        self.hyperparam_space = hyperparam_space
        self._validate_hyperparam_space()

        # ML Evaluator
        # if metrics is None:
        #     metrics = {"accuracy": accuracy_score}
        self.genetic_params = genetic_params

        # State vars
        self.eval_dict = {}
        self.populations = []
        self.logbook = None
        self.mlopt_seed = seed
        self.set_mlopt_seed(seed)

        # Parallel
        self.use_parallel = use_parallel

        # mlflow
        self.use_mlflow = use_mlflow

        # File output control
        self.disable_file_output = disable_file_output

        # Early stopping
        self.early_stopping = early_stopping
        if self.early_stopping:
            if patience <= 0:
                raise ValueError("Patience must be a positive integer.")
            if min_delta < 0:
                raise ValueError("min_delta must be non-negative.")
        self.patience = patience
        self.min_delta = min_delta

        # Initial population seeding
        self.initial_params = initial_params
        self.include_default = include_default

        # Tracker
        self.tracker = Tracker(name="mloptimizer", folder=folder, use_mlflow=self.use_mlflow,
                               use_parallel=self.use_parallel, disable_file_output=self.disable_file_output)

        # Evaluator
        self.individual_utils = IndividualUtils(hyperparam_space=self.hyperparam_space,
                                                estimator_class=self.estimator_class, mlopt_seed=self.mlopt_seed)
        self.evaluator = Evaluator(estimator_class=self.estimator_class, features=features, labels=labels,
                                   eval_function=eval_function, fitness_score=fitness_score,
                                   metrics=metrics, tracker=self.tracker,
                                   individual_utils=self.individual_utils,
                                   seed=self.mlopt_seed)

        # DeapOptimizer
        # self.deap_optimizer = None
        self.genetic_algorithm = None
        #self.runs = []

    def _validate_hyperparam_space(self):
        """
        Method to validate the hyperparam_space
        """
        if self.hyperparam_space is None:
            raise ValueError("hyperparam_space is None")

        # Get default params of the estimator_class
        default_estimator_params = set(self.estimator_class().get_params().keys())

        # For estimators like CatBoost that don't properly implement get_params(),
        # use inspect to get constructor parameters
        is_catboost = 'catboost' in self.estimator_class.__module__.lower()
        if not default_estimator_params or is_catboost:
            sig = inspect.signature(self.estimator_class.__init__)
            default_estimator_params = set(
                param_name for param_name in sig.parameters.keys()
                if param_name != 'self'
            )

        hyperparam_space_params = set(self.hyperparam_space.get_all_params().keys())
        illegal_params = hyperparam_space_params - default_estimator_params

        if len(illegal_params) > 0:
            raise ValueError(f"Parameters {illegal_params} are not parameters of {self.estimator_class.__name__}")

        # Check values of the parameters

    def set_mlopt_seed(self, seed):
        """
        Method to set the seed for the random functions

        Parameters
        ----------
        seed : int
            seed for the random functions
        """
        self.mlopt_seed = seed
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def get_subclasses(my_class):
        """
        Method to get all the subclasses of a class
        (in this case use to get all the classifiers that can be optimized).

        Parameters
        ----------
        my_class : class
            class to get the subclasses

        Returns
        -------
        list
            list of subclasses
        """
        subclasses = my_class.__subclasses__()
        if len(subclasses) == 0:
            return []
        next_subclasses = []
        [next_subclasses.extend(Optimizer.get_subclasses(x)) for x in subclasses]
        return [*subclasses, *next_subclasses]

    def get_clf(self, individual):
        # individual_dict = self.deap_optimizer.individual2dict(individual)
        # clf = self.estimator_class(random_state=self.mlopt_seed, **individual_dict)
        return self.individual_utils.get_clf(individual)

    #def optimize_clf(self, population_size: int = 10, generations: int = 3,
    #                 cxpb=0.5, mutpb=0.5, tournsize=4, indpb=0.5, n_elites=10,
    #                 checkpoint: str = None, opt_run_folder_name: str = None) -> object:
    def optimize_clf(self, population_size: int = 10, generations: int = 3,
                     cxpb=0.5, mutpb=0.5, tournsize=4, indpb=0.5, n_elites=10,
                     checkpoint: str = None, opt_run_folder_name: str = None) -> object:
        """
        Method to optimize the classifier. It uses the custom_ea_simple method to optimize the classifier.

        Parameters
        ----------
        population_size : int, optional (default=10)
            number of individuals in each generation
        generations : int, optional (default=3)
            number of generations
        cxpb : float, optional (default=0.5)
            crossover probability
        mutpb : float, optional (default=0.5)
            mutation probability
        tournsize : int, optional (default=4)
            number of individuals to select in the tournament
        indpb : float, optional (default=0.5)
            independent probability for each attribute to be mutated
        n_elites : int, optional (default=10)
            number of elites to keep in the next generation
        checkpoint : str, optional (default=None)
            path to the checkpoint file
        opt_run_folder_name : str, optional (default=None)
            name of the folder where the execution will be saved

        Returns
        -------
        clf : classifier
            classifier with the best hyperparams
        """
        #from mloptimizer.domain.optimization import DeapOptimizer, GeneticAlgorithmRunner
        from mloptimizer.domain.optimization import GeneticAlgorithm
        # Log initialization with enhanced info
        self.tracker.start_optimization(
            type(self).__name__,
            generations=generations,
            population_size=population_size,
            estimator_class=self.estimator_class
        )

        # Creation of folders and checkpoint
        self.tracker.start_checkpoint(opt_run_folder_name, estimator_class=self.estimator_class)

        # Log dataset
        self.tracker.log_dataset(self.features, self.labels)

        # Log genetic parameters
        self.tracker.log_genetic_params(self.genetic_params)

        # Log comprehensive optimization configuration (Phase 1 MLflow improvement)
        if self.tracker.use_mlflow:
            config = {
                'estimator_class': self.estimator_class.__name__,
                'population_size': population_size,
                'generations': generations,
                'early_stopping': self.early_stopping,
                'patience': self.patience if self.early_stopping else None,
                'min_delta': self.min_delta if self.early_stopping else None,
                'use_parallel': self.use_parallel,
                'n_evolvable_params': len(self.hyperparam_space.evolvable_hyperparams),
                'evolvable_params': list(self.hyperparam_space.evolvable_hyperparams.keys()),
            }
            self.tracker.log_optimization_config(config)

            # Set comprehensive tags (Phase 1 MLflow improvement)
            tags = {
                'estimator_class': self.estimator_class.__name__,
                'mloptimizer_version': get_version(),
                'optimization_algorithm': 'genetic_algorithm',
            }
            self.tracker.set_optimization_tags(tags)

        # Creation of deap optimizer
        #self.deap_optimizer = DeapOptimizer(hyperparam_space=self.hyperparam_space, seed=self.mlopt_seed,
        #                                    use_parallel=self.use_parallel,
        #                                    maximize=is_classifier(self.estimator_class))
        # Creation of genetic algorithm runner
        #ga_runner = GeneticAlgorithmRunner(deap_optimizer=self.deap_optimizer,
        #                                   tracker=self.tracker,
        #                                   seed=self.mlopt_seed,
        #                                   evaluator=self.evaluator)
        self.genetic_algorithm = GeneticAlgorithm(hyperparam_space=self.hyperparam_space,
                                                  tracker=self.tracker,
                                                  seed=self.mlopt_seed,
                                                  evaluator=self.evaluator,
                                                  use_parallel=self.use_parallel,
                                                  maximize=is_classifier(self.estimator_class),
                                                  estimator_class=self.estimator_class)


        # Run genetic algorithm
        population, logbook, hof = self.genetic_algorithm.custom_run(population_size=population_size,
                                                                     n_generations=generations,
                                                                     cxpb=cxpb, mutation_prob=mutpb,
                                                                     n_elites=n_elites,
                                                                     tournsize=tournsize,
                                                                     indpb=indpb,
                                                                     checkpoint=checkpoint,
                                                                     early_stopping=self.early_stopping,
                                                                     patience=self.patience,
                                                                     min_delta=self.min_delta,
                                                                     initial_params=self.initial_params,
                                                                     include_default=self.include_default)

        #self.runs.append(ga_runner)
        #self.logbook = logbook

        # self.populations = population
        # self.populations.append([[ind, ind.fitness] for ind in population])

        # Log and save results
        # self._log_and_save_results(hof)

        # Calculate final statistics
        best_fitness = hof[0].fitness.values[0] if hof else None
        total_evaluations = sum(record['nevals'] for record in logbook) if logbook else None

        # End optimization with comprehensive summary (Phase 1 MLflow improvement)
        self.tracker.end_optimization(
            best_fitness=best_fitness,
            total_evaluations=total_evaluations,
            stopped_early=self.genetic_algorithm.stopped_early_,
            stopped_at_generation=self.genetic_algorithm.generations_run_
        )

        return self.individual_utils.get_clf(hof[0])
