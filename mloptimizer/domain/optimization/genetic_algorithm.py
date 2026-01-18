import logging
import random
import os
import copy
import joblib
import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms
from deap.algorithms import eaSimple, varAnd

from mloptimizer.domain.hyperspace import HyperparameterSpace

logger = logging.getLogger(__name__)
from matplotlib import pyplot as plt
from mloptimizer.application.reporting.plots import plotly_search_space, plotly_logbook, plot_logbook
from mloptimizer.infrastructure.tracking import Tracker
from mloptimizer.domain.evaluation import Evaluator
from mloptimizer.domain.population import IndividualUtils

def mutGaussianInt(individual, low, up, indpb, sigma=0.3):
    """
    Gaussian mutation for integer-encoded individuals with bounds.

    Unlike mutUniformInt (which does random replacement), this mutation
    makes perturbations around the current value. Uses Gaussian distribution
    so small changes are more likely but large jumps are still possible.

    Parameters
    ----------
    individual : list
        The individual to mutate
    low : list
        Lower bounds for each gene
    up : list
        Upper bounds for each gene
    indpb : float
        Probability of mutating each gene
    sigma : float
        Standard deviation as fraction of range (default 0.3 = 30% of range)

    Returns
    -------
    tuple
        A tuple containing the mutated individual
    """
    for i, (xl, xu) in enumerate(zip(low, up)):
        if random.random() < indpb:
            range_size = xu - xl
            if range_size <= 0:
                continue
            # Gaussian perturbation: small changes more likely, large still possible
            delta = int(round(random.gauss(0, sigma * range_size)))
            # Ensure at least some change when mutation occurs
            if delta == 0:
                delta = random.choice([-1, 1])
            new_val = individual[i] + delta
            # Clamp to bounds
            individual[i] = max(xl, min(xu, new_val))
    return individual,

class GeneticAlgorithm:
    def __init__(self, hyperparam_space: HyperparameterSpace = None, tracker: Tracker = None,
                 seed=None, evaluator: Evaluator=None, use_parallel=False, maximize=True,
                 estimator_class=None):
        """
        Class to run the genetic algorithm (combined functionality from DeapOptimizer and GeneticAlgorithmRunner)

        Parameters
        ----------
        hyperparam_space : HyperparameterSpace
            Hyperparameter space for optimization
        tracker : Tracker
            Tracker for logging and checkpointing
        seed : int
            Seed for random operations
        evaluator : Evaluator
            Function to evaluate an individual
        use_parallel : bool
            Flag for parallel processing
        maximize : bool
            Whether to maximize or minimize the objective function
        estimator_class : class
            The estimator class being optimized (used for default individual seeding)
        """
        self.hyperparam_space = hyperparam_space
        self.tracker = tracker
        self.evaluator = evaluator
        self.use_parallel = use_parallel
        self.maximize = maximize
        self.seed = seed
        self.estimator_class = estimator_class
        random.seed(seed)
        np.random.seed(seed)
        self.populations = []

        # Toolbox setup (from DeapOptimizer)
        self.toolbox = base.Toolbox()
        self.eval_dict = {}
        self.logbook = None
        self.stats = None
        self.setup()

        self.generations_run_ = 0
        self.stopped_early_ = False

    def setup(self):
        """
        Set up DEAP's toolbox and statistics (from DeapOptimizer)
        """
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.stats.register("med", np.median)

        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        if self.maximize:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        else:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

        if self.use_parallel:
            try:
                from joblib import Parallel, delayed

                def joblib_map(func, iterable):
                    """
                    Wrapper to make joblib.Parallel compatible with DEAP's toolbox.map.
                    Uses loky backend which can handle closures and complex objects.

                    Parameters
                    ----------
                    func : callable
                        Function to apply to each item
                    iterable : iterable
                        Items to process

                    Returns
                    -------
                    list
                        Results from applying func to each item
                    """
                    # Use loky backend for better pickling support (can handle closures)
                    # n_jobs=-1 uses all available CPU cores
                    # verbose=0 suppresses joblib progress messages
                    return Parallel(n_jobs=-1, backend='loky', verbose=0)(
                        delayed(func)(item) for item in iterable
                    )

                self.toolbox.register("map", joblib_map)
                if self.tracker:
                    logger.debug("Parallelization enabled using joblib with loky backend")
            except ImportError:
                print("joblib not available, falling back to sequential execution")
                self.use_parallel = False

        self.toolbox.register("individual",
                              IndividualUtils(hyperparam_space=self.hyperparam_space,
                                              mlopt_seed=self.seed).init_individual,creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluator.evaluate_individual)
        self.toolbox.register("clone", copy.deepcopy)

    def simple_run(self, population_size: int, n_generations: int, cxpb: float = 0.5, mutation_prob: float = 0.5,
                   n_elites: int = 10, tournsize: int = 3, indpb: float = 0.05):
        """
        Run the genetic algorithm using DEAP's eaSimple method (from GeneticAlgorithmRunner)
        """
        hof, pop = self._pre_run(indpb=indpb, n_elites=n_elites, population_size=population_size, tournsize=tournsize)
        population, logbook = eaSimple(population=pop, toolbox=self.toolbox, cxpb=cxpb, mutpb=mutation_prob,
                                       ngen=n_generations, stats=self.stats, halloffame=hof, verbose=True)
        return population, logbook, hof

    def _pre_run(self, indpb: float = 0.5, n_elites: int = 10, population_size: int = 10, tournsize: int = 4,
                  initial_params: list = None, include_default: bool = False):
        """
        Initialize the population and hall of fame (from GeneticAlgorithmRunner)

        Parameters
        ----------
        indpb : float
            Independent probability for each attribute to be mutated
        n_elites : int
            Number of elites in hall of fame
        population_size : int
            Size of the population
        tournsize : int
            Tournament size for selection
        initial_params : list of dict, optional
            List of hyperparameter dictionaries to seed the population with
        include_default : bool, optional
            If True, include an individual representing sklearn defaults
        """
        # Create initial individuals from user-provided params and/or defaults
        seed_individuals = []
        individual_utils = IndividualUtils(hyperparam_space=self.hyperparam_space, mlopt_seed=self.seed)

        # Add default individual if requested
        if include_default and self.estimator_class is not None:
            default_ind = individual_utils.get_default_individual(
                self.estimator_class, pcls=creator.Individual
            )
            seed_individuals.append(default_ind)

        # Add user-provided initial params
        if initial_params:
            for params in initial_params:
                ind = individual_utils.params_to_individual(params, pcls=creator.Individual)
                seed_individuals.append(ind)

        # Create remaining random individuals
        n_random = max(0, population_size - len(seed_individuals))
        pop = self.toolbox.population(n=n_random)

        # Combine seeded and random individuals
        pop = seed_individuals + pop

        hof = tools.HallOfFame(n_elites)
        self.toolbox.register("mate", tools.cxTwoPoint)
        # Use Gaussian mutation (small perturbations) instead of Uniform (random replacement)
        # This allows fine-tuning good solutions while still enabling exploration
        self.toolbox.register("mutate", mutGaussianInt,
                              low=[x.min_value for x in self.hyperparam_space.evolvable_hyperparams.values()],
                              up=[x.max_value for x in self.hyperparam_space.evolvable_hyperparams.values()],
                              indpb=indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)
        hist = tools.History()
        self.toolbox.decorate("mate", hist.decorator)
        self.toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)
        return hof, pop

    def _log_and_visualize_results(self, logbook):
        """
        Log and visualize the results of the genetic algorithm.
        """
        # Skip visualization if file output is disabled
        if self.tracker.disable_file_output:
            return

        hyperparam_names = list(self.hyperparam_space.evolvable_hyperparams.keys())
        hyperparam_names.append("fitness")
        population_df = self.population_2_df()
        df = population_df[hyperparam_names]
        # Use lower resolution (40) for smaller HTML files (~500KB vs 4MB)
        g = plotly_search_space(df, kde_resolution=40, max_scatter_points=500)
        g.write_html(os.path.join(self.tracker.graphics_path, "search_space.html"),
                     full_html=False, include_plotlyjs='cdn', include_mathjax=False)
        plt.close()

        g2 = plotly_logbook(logbook, population_df)
        g2.write_html(os.path.join(self.tracker.graphics_path, "logbook.html"),
                     full_html=False, include_plotlyjs='cdn', include_mathjax=False)
        plt.close()

        g3 = plot_logbook(logbook)
        g3.savefig(os.path.join(self.tracker.graphics_path, "logbook_s.png"))
        plt.close()

    def custom_run(self, population_size: int, n_generations: int, cxpb: float = 0.5, mutation_prob: float = 0.5,
                   n_elites: int = 10, tournsize: int = 3, indpb: float = 0.05, checkpoint: str = None,
                   early_stopping: bool = False, patience: int = 10, min_delta: float = 0.01,
                   initial_params: list = None, include_default: bool = False):
        """
        Run the genetic algorithm with tracking of each generation (from GeneticAlgorithmRunner)

        Parameters
        ----------
        population_size : int
            Size of the population
        n_generations : int
            Number of generations to run
        cxpb : float
            Crossover probability
        mutation_prob : float
            Mutation probability
        n_elites : int
            Number of elites in hall of fame
        tournsize : int
            Tournament size for selection
        indpb : float
            Independent probability for each attribute to be mutated
        checkpoint : str
            Path to checkpoint file
        early_stopping : bool
            Whether to use early stopping
        patience : int
            Number of generations without improvement before stopping
        min_delta : float
            Minimum improvement to consider as progress
        initial_params : list of dict, optional
            List of hyperparameter dictionaries to seed the population with
        include_default : bool, optional
            If True, include an individual representing sklearn defaults
        """
        hof, pop = self._pre_run(indpb=indpb, n_elites=n_elites, population_size=population_size, tournsize=tournsize,
                                  initial_params=initial_params, include_default=include_default)
        population, logbook, hof = self.custom_ea_simple(population=pop, toolbox=self.toolbox, cxpb=cxpb,
                                                         mutpb=mutation_prob, ngen=n_generations, halloffame=hof,
                                                         checkpoint_path=self.tracker.opt_run_checkpoint_path,
                                                         stats=self.stats, early_stopping=early_stopping,
                                                         patience=patience, min_delta=min_delta,
                                                         n_elites=n_elites)
        self.logbook = logbook
        self._log_and_visualize_results(logbook)

        return population, logbook, hof

    def custom_ea_simple(self, population: list, toolbox: base.Toolbox, cxpb: float = 0.5, mutpb: float = 0.5,
                         start_gen: int = 0, ngen: int = 4, checkpoint_path: str = None, stats: tools.Statistics = None,
                         halloffame: tools.HallOfFame = None, verbose: bool = True,
                         early_stopping: bool = False, patience: int = 10, min_delta: float = 0.01,
                         n_elites: int = 3):
        """
        Custom evolution algorithm with tracking, checkpointing, and proper elitism.

        Parameters
        ----------
        n_elites : int
            Number of elite individuals to preserve each generation. These individuals
            are copied directly to the next generation without modification.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Early stopping variables
        best_fitness = None
        no_improve = 0

        # Evaluate the initial population first (generation 0)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        c = 1
        self.tracker.start_progress_file(0)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            ind_formatted = IndividualUtils(hyperparam_space=self.hyperparam_space,
                                            mlopt_seed=self.seed).individual2dict(ind)
            self.tracker.append_progress_file(0, ngen, c, len(invalid_ind), ind_formatted, fit)
            c += 1

        halloffame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        # Log generation 0 metrics to MLflow (Phase 1 improvement)
        if self.tracker:
            self.tracker.log_generation_metrics(0, record)

        self.populations.append([[ind, ind.fitness] for ind in population])

        for gen in range(max(1, start_gen), ngen + 1):
            self.tracker.start_progress_file(gen)

            # ELITISM: Preserve the best n_elites individuals
            elites = tools.selBest(population, n_elites)
            # Clone elites to avoid modifying them
            elites = [toolbox.clone(ind) for ind in elites]

            # Select parents for offspring (excluding elites spots)
            n_offspring = len(population) - n_elites
            offspring = toolbox.select(population, n_offspring)
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Apply crossover and mutation to offspring
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            # Combine elites with offspring
            population = elites + offspring

            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            c = 1
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind_formatted = IndividualUtils(hyperparam_space=self.hyperparam_space,
                                                mlopt_seed=self.seed).individual2dict(ind)
                self.tracker.append_progress_file(gen, ngen, c, len(invalid_ind), ind_formatted, fit)
                c += 1

            halloffame.update(population)
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # Log generation metrics to MLflow (Phase 1 improvement)
            if self.tracker:
                self.tracker.log_generation_metrics(gen, record)

            self.populations.append([[ind, ind.fitness] for ind in population])

            if checkpoint_path and not self.tracker.disable_file_output:
                cp = dict(population=population, generation=gen, halloffame=halloffame, logbook=logbook,
                          rndstate=self.seed)
                cp_file = os.path.join(checkpoint_path, "cp_gen_{}.pkl".format(gen))
                joblib.dump(cp, cp_file)
            self.tracker.write_population_file(self.population_2_df())
            self.tracker.write_logbook_file(logbook)
            self.tracker.log_clfs([], gen, [])

            # Early stopping logic
            gen_best_fitness = max([ind.fitness.values[0] for ind in population])

            if best_fitness is None or gen_best_fitness > best_fitness + min_delta:
                best_fitness = gen_best_fitness
                no_improve = 0
            else:
                no_improve += 1

            if early_stopping and no_improve >= patience:
                self.generations_run_ = gen
                logger.info("="*70)
                logger.info(f"⚠️  Early Stopping Triggered")
                logger.info(f"  Generation: {gen}/{ngen}")
                logger.info(f"  Best fitness: {best_fitness:.6f}")
                logger.info(f"  No improvement for {no_improve} generations (patience={patience})")
                logger.info(f"  Stopping optimization early to avoid wasted evaluations")
                logger.info("="*70)
                self.stopped_early_ = True
                break

        if not self.stopped_early_:
            self.generations_run_ = ngen

        return population, logbook, halloffame

    def population_2_df(self):
        """
        Convert the population to a pandas dataframe (from GeneticAlgorithmRunner)
        """
        data = []
        n = 0
        for p in self.populations:
            for i in p:
                i_hyperparams = IndividualUtils(hyperparam_space=self.hyperparam_space,
                                                mlopt_seed=self.seed).individual2dict(i[0])
                i_hyperparams['fitness'] = i[1].values[0]
                i_hyperparams['population'] = n
                data.append(i_hyperparams)
            n += 1

        return pd.DataFrame(data)
