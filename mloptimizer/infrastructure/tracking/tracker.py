import logging
import os
import shutil
from datetime import datetime
import importlib
import joblib
import pandas as pd
import tqdm

from mloptimizer.infrastructure.util.utils import create_optimization_folder

logger = logging.getLogger(__name__)


class Tracker:
    """
    Tracker class for logging and tracking the optimization process.

    Parameters
    ----------
    name : str
        Name of the optimization process.
    folder : str
        Folder where the optimization process will be stored.
    use_parallel : bool
        If True, the optimization process will be executed in parallel. Default is False.
        The use of parallel execution is not compatible with tqdm, no progress bar will be shown.
        Also, XGBClassifier is not compatible with parallel execution.
    use_mlflow : bool
        If True, the optimization process will be tracked using MLFlow. Default is False.
    """

    def __init__(self, name, folder=os.curdir,
                 use_parallel=False, use_mlflow=False, disable_file_output=False):

        self.name = name
        self.gen = 0
        self.individual_index = 0
        self.disable_file_output = disable_file_output

        # Main folder setup
        if not self.disable_file_output:
            # Normal mode: create folder for results
            self.folder = create_optimization_folder(folder)
        else:
            # No file output: skip folder creation
            self.folder = None

        # Paths
        self.opt_run_folder = None
        self.opt_run_checkpoint_path = None
        self.progress_path = None
        self.progress_path = None
        self.results_path = None
        self.graphics_path = None

        # tqdm is not compatible with parallel execution
        self.use_parallel = use_parallel

        if not self.use_parallel:
            self.gen_pbar = None

        # MLFlow
        self.use_mlflow = use_mlflow
        self.parent_run = None  # MLflow parent run for the full optimization

        # Best fitness
        self.best_fitness = None

        if self.use_mlflow:
            try:
                self.mlflow = importlib.import_module("mlflow")
            except ImportError as e:
                error_msg = (
                    "\n" + "="*70 + "\n"
                    "ERROR: MLflow not installed but use_mlflow=True\n"
                    "="*70 + "\n"
                    "MLflow is required when use_mlflow=True but is not installed.\n\n"
                    "To install MLflow, run:\n"
                    "  pip install mlflow\n\n"
                    "Or disable MLflow tracking:\n"
                    "  GeneticSearch(..., use_mlflow=False)\n"
                    + "="*70
                )
                logger.error(error_msg)
                raise ImportError(
                    "MLflow is required when use_mlflow=True. "
                    "Install it with: pip install mlflow"
                ) from e

    def start_optimization(self, opt_class, generations: int, population_size=None, estimator_class=None):
        """
        Start the optimization process.

        Parameters
        ----------
        opt_class : str
            Name of the optimization class.
        generations : int
            Number of generations for the optimization process.
        population_size : int, optional
            Size of the population for the genetic algorithm.
        estimator_class : class, optional
            The estimator class being optimized.
        """
        # Store for later use in logging and MLflow
        self.total_generations = generations
        self.population_size = population_size or 0
        self.estimator_class = estimator_class
        self.optimization_start_time = datetime.now()

        # Inform the user that the optimization is starting with detailed info
        logger.info("="*70)
        logger.info("Starting Genetic Algorithm Optimization")
        logger.info("="*70)
        logger.info(f"  Optimizer: {opt_class}")
        if estimator_class:
            logger.info(f"  Estimator: {estimator_class.__name__}")
        if population_size:
            logger.info(f"  Population size: {population_size}")
        logger.info(f"  Max generations: {generations}")
        if self.use_parallel:
            logger.info(f"  Parallelization: Enabled (joblib with loky backend)")
        else:
            logger.info(f"  Parallelization: Disabled (sequential execution)")
        if self.use_mlflow:
            logger.info(f"  MLflow tracking: Enabled")
        logger.info("="*70)

        # tqdm is not compatible with parallel execution
        if not self.use_parallel:
            self._init_progress_bar(generations)


    def end_optimization(self, best_fitness=None, total_evaluations=None, stopped_early=False, stopped_at_generation=None):
        """
        Ends the optimization process by finalizing any active MLflow runs.

        Parameters
        ----------
        best_fitness : float, optional
            Best fitness achieved during optimization
        total_evaluations : int, optional
            Total number of evaluations performed
        stopped_early : bool, optional
            Whether early stopping was triggered
        stopped_at_generation : int, optional
            Generation at which optimization stopped

        Raises
        ------
        Exception
            If there is an error while ending the MLflow run.
        """
        # Calculate optimization time
        if hasattr(self, 'optimization_start_time'):
            duration = (datetime.now() - self.optimization_start_time).total_seconds()
        else:
            duration = None

        # Log final summary
        logger.info("="*70)
        logger.info("Optimization Complete")
        logger.info("="*70)
        if best_fitness is not None:
            logger.info(f"  Best fitness achieved: {best_fitness:.6f}")
        if total_evaluations is not None:
            logger.info(f"  Total evaluations: {total_evaluations}")
        if stopped_early:
            logger.info(f"  Early stopping: Yes (stopped at generation {stopped_at_generation})")
        if duration is not None:
            logger.info(f"  Optimization time: {duration:.2f} seconds")
        logger.info("="*70)

        # Log final tags to MLflow
        if self.use_mlflow and best_fitness is not None:
            try:
                final_tags = {}
                if stopped_early:
                    final_tags['early_stopped'] = 'True'
                    if stopped_at_generation is not None:
                        final_tags['stopped_at_generation'] = str(stopped_at_generation)
                if total_evaluations is not None:
                    final_tags['total_evaluations'] = str(total_evaluations)
                if duration is not None:
                    final_tags['optimization_time_seconds'] = f"{duration:.2f}"

                if final_tags:
                    self.mlflow.set_tags(final_tags)

                # Log final metric
                self.mlflow.log_metric('final_best_fitness', best_fitness)

            except Exception as e:
                logger.warning(f"Failed to log final tags/metrics: {e}")

        # Close MLflow run
        if self.use_mlflow and self.parent_run is not None:
            try:
                self.mlflow.end_run()
                logger.debug("MLflow run closed successfully")
            except Exception as e:
                logger.exception("Error closing MLflow run: %s", e)
            finally:
                self.parent_run = None

    def start_mlflow_experiment(self):
        """
        Start a new MLflow experiment with the specified run name.

        Parameters
        ----------
        run_name : str
            Name of the MLflow experiment to be started.
        """
        try:
            try:
                self.mlflow.set_experiment(self.name)
                exp = self.mlflow.get_experiment_by_name(self.name)
                if exp is not None:
                    logger.debug("MLflow Experiment: '%s' (ID: %s)", exp.name, exp.experiment_id)
                else:
                    logger.warning("MLflow experiment '%s' could not be retrieved", self.name)
            except Exception as e:
                logger.exception("Failed to initialize MLflow experiment '%s': %s", self.name, e)
        except Exception as fatal:
            logger.exception("Critical error starting MLflow experiment: %s", fatal)
            raise

    def start_mlflow_run(self, run_name: str):
        """
        Start a new MLflow run with the specified run name.

        Parameters
        ----------
        run_name : str
            Name of the MLflow run to be started.
        """
        try:
            active = None
            try:
                active = self.mlflow.active_run()
            except Exception as e:
                logger.warning("Could not read active MLflow run: %s", e)

            if active is not None:
                try:
                    logger.warning(
                        "MLflow active run detected before starting new run: run_id=%s, experiment_id=%s",
                        active.info.run_id, active.info.experiment_id
                    )
                except Exception:
                    logger.warning("MLflow active run detected (details unreadable)")

                # Intentar cerrar el run activo para evitar colisiones
                try:
                    self.mlflow.end_run()
                    logger.info("Closed previous MLflow active run successfully.")
                except Exception as e:
                    logger.exception("Failed to end active MLflow run: %s. Will attempt nested run.", e)

            # Intentar iniciar un nuevo run normalmente
            try:
                self.parent_run = self.mlflow.start_run(run_name=run_name)
                logger.debug("MLflow Run Started: '%s' (ID: %s)", run_name, self.parent_run.info.run_id)
            except Exception as e:
                logger.exception("Failed to start MLflow run, retrying with nested=True: %s", e)
                # Reintentar como nested run para no chocar con runs globales que no se pueden cerrar
                self.parent_run = self.mlflow.start_run(run_name=run_name, nested=True)
                logger.debug("MLflow Nested Run Started: '%s' (ID: %s)", run_name, self.parent_run.info.run_id)

        except Exception as fatal:
            logger.exception("Unexpected error when starting MLflow checkpoint: %s", fatal)
            raise

    def start_checkpoint(self, opt_run_folder_name, estimator_class):
        """
        Start a checkpoint for the optimization process.

        Parameters
        ----------
        opt_run_folder_name : str
            Name of the folder where the checkpoint will be stored. (not the full path)
        estimator_class : class
            Class of the estimator being optimized used to create the checkpoint path.
        """
        # Skip all directory creation if file output is disabled
        if self.disable_file_output:
            # Set paths to None to prevent file operations later
            self.opt_run_folder = None
            self.opt_run_checkpoint_path = None
            self.results_path = None
            self.graphics_path = None
            self.progress_path = None

            if self.use_mlflow:
                self.start_mlflow_experiment()
                # Generate a run name even without folders
                if not opt_run_folder_name:
                    opt_run_folder_name = "{}_{}".format(
                        datetime.now().strftime("%Y%m%d_%H%M%S"),
                        estimator_class.__name__)
                self.start_mlflow_run(opt_run_folder_name)
            return

        # Create checkpoint_path from date and algorithm
        if not opt_run_folder_name:
            opt_run_folder_name = "{}_{}".format(
                datetime.now().strftime("%Y%m%d_%H%M%S"),
                estimator_class.__name__)

        self.opt_run_folder = os.path.join(self.folder, opt_run_folder_name)
        self.opt_run_checkpoint_path = os.path.join(self.opt_run_folder, "checkpoints")
        self.results_path = os.path.join(self.opt_run_folder, "results")
        self.graphics_path = os.path.join(self.opt_run_folder, "graphics")
        self.progress_path = os.path.join(self.opt_run_folder, "progress")

        if os.path.exists(self.opt_run_folder):
            shutil.rmtree(self.opt_run_folder)
        os.mkdir(self.opt_run_folder)
        os.mkdir(self.opt_run_checkpoint_path)
        os.mkdir(self.results_path)
        os.mkdir(self.graphics_path)
        os.mkdir(self.progress_path)

        if self.use_mlflow:
            self.start_mlflow_experiment()
            self.start_mlflow_run(opt_run_folder_name)


    def log_dataset(self, X, y):
        """Log dataset information to MLflow with comprehensive metadata."""
        if self.use_mlflow:
            try:
                # Log dataset
                df_dataset = pd.DataFrame(X)
                df_dataset["label"] = y
                dataset = self.mlflow.data.from_pandas(df_dataset)
                self.mlflow.log_input(dataset, context="training")

                # Log dataset metadata as tags
                import numpy as np
                n_samples, n_features = X.shape
                n_classes = len(np.unique(y)) if y is not None else 0

                dataset_tags = {
                    'dataset_samples': str(n_samples),
                    'dataset_features': str(n_features),
                    'dataset_classes': str(n_classes) if n_classes > 0 else 'regression',
                }
                self.mlflow.set_tags(dataset_tags)

                logger.debug(
                    f"Dataset logged: {n_samples} samples, {n_features} features"
                    + (f", {n_classes} classes" if n_classes > 1 else "")
                )

            except Exception as e:
                logger.warning(f"Failed to log dataset: {e}")

    def log_clfs(self, classifiers_list: list, generation: int, fitness_list: list[float]):
        self.gen = generation
        self.individual_index = 0
        for i in range(len(classifiers_list)):
            logger.info(f"Generation {generation} - Classifier TOP {i}")
            logger.info(f"Classifier: {classifiers_list[i]}")
            logger.info(f"Fitness: {fitness_list[i]}")
            logger.info("Hyperparams: {}".format(str(classifiers_list[i].get_params())))
        self.gen = generation + 1

    def log_evaluation(self, classifier, metrics, fitness_score, greater_is_better=True):
        # tqdm is not compatible with parallel execution
        if not self.use_parallel:
            # Update best fitness and progress bar postfix
            need_update = self.best_fitness is None or ((greater_is_better and self.best_fitness < fitness_score) or
                                                        (not greater_is_better and self.best_fitness > fitness_score))
            if need_update:
                self.best_fitness = fitness_score
                self.gen_pbar.set_postfix({"best fitness": self.best_fitness})

        logger.debug(f"Adding to mlflow...\nClassifier: {classifier}\nMetrics: {metrics}")
        self.individual_index += 1
        individual_index = self.individual_index
        if self.use_mlflow:
            run_name = f"gen_{self.gen}_ind_{individual_index}_{classifier.__class__.__name__}"
            with self.mlflow.start_run(run_name=run_name, nested=True):
                self.mlflow.log_params(classifier.get_params())
                # We use the generation as the step
                # self.mlflow.log_metric(key="fitness", value=metric, step=self.gen)
                self.mlflow.log_metrics(metrics, step=self.gen)
                self.mlflow.set_tag("generation", self.gen)
                self.mlflow.set_tag("individual_index", individual_index)
                self.mlflow.set_tag("estimator", classifier.__class__.__name__)

    def load_checkpoint(self, checkpoint):

        # Extract checkpoint_path from checkpoint file
        self.opt_run_checkpoint_path = os.path.dirname(checkpoint)
        self.opt_run_folder = os.path.dirname(self.opt_run_checkpoint_path)
        logger.debug("Initiating from checkpoint {}...".format(checkpoint))

        self.results_path = os.path.join(self.opt_run_folder, "results")
        self.graphics_path = os.path.join(self.opt_run_folder, "graphics")
        self.progress_path = os.path.join(self.opt_run_folder, "progress")
        cp = joblib.load(checkpoint)
        return cp

    def write_logbook_file(self, logbook, filename=None):
        """
        Method to write the logbook to a csv file

        Parameters
        ----------
        logbook : ~deap.tools.Logbook
            logbook of the optimization process
        filename : str, optional (default=None)
            filename to save the logbook
        """
        if self.disable_file_output:
            return
        if filename is None:
            filename = os.path.join(self.results_path, 'logbook.csv')
        pd.DataFrame(logbook).to_csv(filename, index=False)

    def write_population_file(self, populations: pd.DataFrame, filename=None):
        """
        Method to write the population to a csv file

        Parameters
        ----------
        populations: pd.DataFrame
            population of the optimization process
        filename : str, optional (default=None)
            filename to save the population
        """
        if self.disable_file_output:
            return
        if filename is None:
            filename = os.path.join(self.results_path, 'populations.csv')
        populations.sort_values(by=['fitness'], ascending=False
                                ).to_csv(filename, index=False)

    def start_progress_file(self, gen: int):
        # tqdm is not compatible with parallel execution
        if not self.use_parallel:
            self.gen_pbar.update()

        if self.disable_file_output:
            return

        progress_gen_path = os.path.join(self.progress_path, "Generation_{}.csv".format(gen))
        header_progress_gen_file = "i;total;Individual;fitness\n"
        with open(progress_gen_path, "w") as progress_gen_file:
            progress_gen_file.write(header_progress_gen_file)
            progress_gen_file.close()

        logger.debug("Generation: {}".format(gen))

        # self.pbar.refresh()

    def append_progress_file(self, gen: int, ngen: int, c: int, evaluations_pending: int, ind_formatted, fit):
        logger.debug(
            "Fitting individual (informational purpose): gen {} - ind {} of {}".format(
                gen, c, evaluations_pending
            )
        )

        if self.disable_file_output:
            if not self.use_parallel and gen == ngen and c == evaluations_pending:
                self.gen_pbar.close()
            return

        progress_gen_path = os.path.join(self.progress_path, "Generation_{}.csv".format(gen))
        with open(progress_gen_path, "a") as progress_gen_file:
            progress_gen_file.write(
                "{};{};{};{}\n".format(c,
                                       evaluations_pending,
                                       ind_formatted, fit)
            )
        if not self.use_parallel and gen == ngen and c == evaluations_pending:
            self.gen_pbar.close()

    def _init_progress_bar(self, n_generations, msg="Genetic execution"):
        self.gen_pbar = tqdm.tqdm(desc=msg, total=n_generations+1, postfix={"best fitness": "?"})
        # self.pbar.refresh()

    def log_genetic_params(self, genetic_params):
        """Log genetic algorithm parameters to MLflow."""
        if self.use_mlflow:
            # Log all genetic params
            self.mlflow.log_params(genetic_params)
            logger.debug(f"Logged {len(genetic_params)} genetic algorithm parameters to MLflow")

    def log_generation_metrics(self, generation, stats_record):
        """
        Log generation-level metrics to MLflow.

        This implements Phase 1 of the MLflow improvement plan by logging
        population statistics per generation.

        Parameters
        ----------
        generation : int
            Current generation number
        stats_record : dict
            Statistics for this generation (min, max, avg, std, etc.)
        """
        if not self.use_mlflow:
            return

        try:
            metrics = {}

            # Core fitness metrics
            if 'max' in stats_record:
                metrics['generation_best_fitness'] = stats_record['max']
            if 'avg' in stats_record:
                metrics['generation_avg_fitness'] = stats_record['avg']
            if 'min' in stats_record:
                metrics['generation_worst_fitness'] = stats_record['min']
            if 'std' in stats_record:
                metrics['generation_fitness_std'] = stats_record['std']
            if 'med' in stats_record:
                metrics['generation_median_fitness'] = stats_record['med']

            # Log with generation as step for time-series view
            if metrics:
                self.mlflow.log_metrics(metrics, step=generation)

                # Log progress message
                best = stats_record.get('max', 0)
                avg = stats_record.get('avg', 0)
                logger.info(
                    f"Generation {generation}/{self.total_generations}: "
                    f"Best={best:.4f}, Avg={avg:.4f}, "
                    f"Evaluations={stats_record.get('nevals', 0)}"
                )

        except Exception as e:
            logger.warning(f"Failed to log generation metrics: {e}")

    def log_optimization_config(self, config_dict):
        """
        Log comprehensive optimization configuration to MLflow.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary with all optimization settings
        """
        if not self.use_mlflow:
            return

        try:
            # Flatten nested config if needed and log as params
            flat_config = {}
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_config[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_config[key] = value

            self.mlflow.log_params(flat_config)
            logger.debug(f"Logged {len(flat_config)} configuration parameters to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log configuration: {e}")

    def set_optimization_tags(self, tags_dict):
        """
        Set comprehensive tags for the optimization run.

        Parameters
        ----------
        tags_dict : dict
            Dictionary of tags to set for this run
        """
        if not self.use_mlflow:
            return

        try:
            self.mlflow.set_tags(tags_dict)
            logger.debug(f"Set {len(tags_dict)} MLflow tags")
        except Exception as e:
            logger.warning(f"Failed to set tags: {e}")

    def info(self, msg):
        """
        Log an informational message.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        logger.info(msg)
