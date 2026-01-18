from mloptimizer.infrastructure.repositories import HyperparameterSpaceRepository
from mloptimizer.domain.hyperspace import HyperparameterSpace
import os


class HyperparameterSpaceService:
    """
    Service to manage hyperparameter spaces.
    """
    base_config_path:  str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "infrastructure", "config", "hyperspace")

    @staticmethod
    def _detect_library(estimator_class):
        module_name = estimator_class.__module__
        supported_modules = ['sklearn', 'catboost', 'xgboost', 'lightgbm']

        for module in supported_modules:
            if module_name.startswith(module):
                return module

        raise ValueError(f"Estimator class {estimator_class.__name__} not supported")

    def load_default_hyperparameter_space(self, estimator_class):
        """
        Load a default hyperparameter space for an estimator from a JSON file.
        """
        file_name = f"{estimator_class.__name__}_default_HyperparamSpace.json"
        library = self._detect_library(estimator_class)
        file_path = os.path.join(self.base_config_path, library, file_name)

        return self.load_hyperparameter_space(file_path)

    @staticmethod
    def load_hyperparameter_space(hyperparam_space_json_path):
        """
        Load a hyperparameter space for an estimator from a JSON file.
        """
        if not os.path.exists(hyperparam_space_json_path):
            raise FileNotFoundError(f"The file {hyperparam_space_json_path} does not exist")

        # Delegate to repository
        hyperparam_data = HyperparameterSpaceRepository.load_json(hyperparam_space_json_path)

        return HyperparameterSpace.from_json_data(hyperparam_data)

    @staticmethod
    def save_hyperparameter_space(hyperparam_space: HyperparameterSpace,
                                  file_path, overwrite=False):
        """
        Save a hyperparameter space for an estimator to a JSON file.
        """
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f"The file {file_path} already exists.")

        # Delegate to repository
        hyperparam_data = {
            'fixed_hyperparams': hyperparam_space.fixed_hyperparams,
            'evolvable_hyperparams': {k: v.__dict__ for k, v in hyperparam_space.evolvable_hyperparams.items()}
        }
        HyperparameterSpaceRepository.save_json(hyperparam_data, file_path, overwrite)
