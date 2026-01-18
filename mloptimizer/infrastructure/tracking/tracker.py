from mloptimizer.infrastructure.util.utils import create_optimization_folder, init_logger
import os
import shutil
from datetime import datetime
import importlib
import joblib
import pandas as pd
import tqdm


class Tracker:
    """
    Tracker class for logging and tracking the optimization process.

    Parameters
    ----------
    name : str
        Name of the optimization process.
    folder : str
        Folder where the optimization process will be stored.
    log_file : str
        Name of the log file.
    use_parallel : bool
        If True, the optimization process will be executed in parallel. Default is False.
        The use of parallel execution is not compatible with tqdm, no progress bar will be shown.
        Also, XGBClassifier is not compatible with parallel execution.
    use_mlflow : bool
        If True, the optimization process will be tracked using MLFlow. Default is False.
    """

    def __init__(self, name, folder=os.curdir, log_file="mloptimizer.log",
                 use_parallel=False, use_mlflow=False, disable_file_output=False):

        self.name = name
        self.gen = 0
        self.individual_index = 0
        self.disable_file_output = disable_file_output

        # Main folder and logger setup
        if not self.disable_file_output:
            # Normal mode: create folder and log to file
            self.folder = create_optimization_folder(folder)
            self.mloptimizer_logger, self.log_file = init_logger(log_file, self.folder, file_output=True)
        else:
            # No file output: skip folder creation, log only to console
            self.folder = None
            self.mloptimizer_logger, self.log_file = init_logger(log_file, file_output=False)
        self.optimization_logger = None

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
            self.mlflow = importlib.import_module("mlflow")

    def start_optimization(self, opt_class, generations: int):
        """
        Start the optimization process.

        Parameters
        ----------
        opt_class : str
            Name of the optimization class.
        generations : int
            Number of generations for the optimization process.
        """
        # Inform the user that the optimization is starting
        self.mloptimizer_logger.info(f"Initiating genetic optimization...")
        # self.mloptimizer_logger.info("Algorithm: {}".format(type(self).__name__))
        self.mloptimizer_logger.info(f"Algorithm: {opt_class}")

        # tqdm is not compatible with parallel execution
        if not self.use_parallel:
            self._init_progress_bar(generations)


    def end_optimization(self):
        """
        Ends the optimization process by finalizing any active MLflow runs.

        Raises
        ------
        Exception
            If there is an error while ending the MLflow run.
        """
        if self.use_mlflow and self.parent_run is not None:
            try:
                self.mlflow.end_run()
                self.optimization_logger.info("Closed parent MLflow run.")
            except Exception as e:
                self.optimization_logger.exception("Error closing parent MLflow run: %s", e)
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
                    self.optimization_logger.info("Using MLflow experiment: %s id=%s", exp.name, exp.experiment_id)
                else:
                    self.optimization_logger.warning("MLflow set_experiment did not return an experiment for %s", self.name)
            except Exception as e:
                self.optimization_logger.exception("Failed to set/get MLflow experiment %s: %s", self.name, e)
        except Exception as fatal:
            self.optimization_logger.exception("Unexpected error when starting MLflow checkpoint: %s", fatal)
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
                self.optimization_logger.warning("Could not read active MLflow run: %s", e)

            if active is not None:
                try:
                    self.optimization_logger.warning(
                        "MLflow active run detected before starting new run: run_id=%s, experiment_id=%s",
                        active.info.run_id, active.info.experiment_id
                    )
                except Exception:
                    self.optimization_logger.warning("MLflow active run detected (details unreadable)")

                # Intentar cerrar el run activo para evitar colisiones
                try:
                    self.mlflow.end_run()
                    self.optimization_logger.info("Closed previous MLflow active run successfully.")
                except Exception as e:
                    self.optimization_logger.exception("Failed to end active MLflow run: %s. Will attempt nested run.", e)

            # Intentar iniciar un nuevo run normalmente
            try:
                self.parent_run = self.mlflow.start_run(run_name=run_name)
                self.optimization_logger.info("Started MLflow run: run_id=%s", self.parent_run.info.run_id)
            except Exception as e:
                self.optimization_logger.exception("start_run failed, retrying with nested=True: %s", e)
                # Reintentar como nested run para no chocar con runs globales que no se pueden cerrar
                self.parent_run = self.mlflow.start_run(run_name=run_name, nested=True)
                self.optimization_logger.info("Started nested MLflow run: run_id=%s", self.parent_run.info.run_id)

        except Exception as fatal:
            self.optimization_logger.exception("Unexpected error when starting MLflow checkpoint: %s", fatal)
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

            # Create console-only logger for optimization tracking
            self.optimization_logger, _ = init_logger("opt.log", file_output=False)

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
        self.optimization_logger, _ = init_logger(
            os.path.join(self.opt_run_folder, "opt.log")
        )

        if self.use_mlflow:
            self.start_mlflow_experiment()
            self.start_mlflow_run(opt_run_folder_name)


    def log_dataset(self, X, y):
        if self.use_mlflow:
            df_dataset = pd.DataFrame(X)
            df_dataset["label"] = y
            dataset = self.mlflow.data.from_pandas(df_dataset)
            self.mlflow.log_input(dataset, context="training")

    def log_clfs(self, classifiers_list: list, generation: int, fitness_list: list[float]):
        self.gen = generation
        self.individual_index = 0
        for i in range(len(classifiers_list)):
            self.optimization_logger.info(f"Generation {generation} - Classifier TOP {i}")
            self.optimization_logger.info(f"Classifier: {classifiers_list[i]}")
            self.optimization_logger.info(f"Fitness: {fitness_list[i]}")
            self.optimization_logger.info("Hyperparams: {}".format(str(classifiers_list[i].get_params())))
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

        if self.optimization_logger:
            self.optimization_logger.debug(f"Adding to mlflow...\nClassifier: {classifier}\nMetrics: {metrics}")
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
        self.optimization_logger, _ = init_logger(os.path.join(self.opt_run_folder,
                                                               f"opt_{os.path.basename(checkpoint)}.log"))
        self.optimization_logger.debug("Initiating from checkpoint {}...".format(checkpoint))

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

        if self.optimization_logger:
            self.optimization_logger.debug("Generation: {}".format(gen))

        # self.pbar.refresh()

    def append_progress_file(self, gen: int, ngen: int, c: int, evaluations_pending: int, ind_formatted, fit):
        if self.optimization_logger:
            self.optimization_logger.debug(
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
        if self.use_mlflow:
            self.mlflow.log_params(genetic_params)

    def info(self, msg):
        """
        Log an informational message.

        Parameters
        ----------
        msg : str
            The message to log.
        """
        self.mloptimizer_logger.info(msg)
        if self.optimization_logger:
            self.optimization_logger.info(msg)
