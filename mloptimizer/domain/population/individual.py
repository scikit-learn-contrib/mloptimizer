from mloptimizer.domain.hyperspace import HyperparameterSpace
from sklearn.svm import SVR
import random


class IndividualUtils:
    def __init__(self, hyperparam_space: HyperparameterSpace = None, estimator_class=None, mlopt_seed=None):
        self.hyperparam_space = hyperparam_space
        self.estimator_class = estimator_class
        self.mlopt_seed = mlopt_seed
        if self.mlopt_seed is not None:
            random.seed(self.mlopt_seed)

    def get_clf(self, individual):
        individual_dict = self.individual2dict(individual)
        if self.estimator_class is SVR:
            # SVR is a deterministic model, so it does not require a random seed
            clf = self.estimator_class(**individual_dict)
        else:
            clf = self.estimator_class(random_state=self.mlopt_seed, **individual_dict)
        return clf

    def individual2dict(self, individual):
        """
        Method to convert an individual to a dictionary of hyperparams

        Parameters
        ----------
        individual : individual
            individual to convert

        Returns
        -------
        individual_dict : dict
            dictionary of hyperparams
        """
        individual_dict = {}
        keys = list(self.hyperparam_space.evolvable_hyperparams.keys())
        for i in range(len(keys)):
            individual_dict[keys[i]] = self.hyperparam_space.evolvable_hyperparams[keys[i]].correct(individual[i])
        return {**individual_dict, **self.hyperparam_space.fixed_hyperparams}

    def init_individual(self, pcls):
        """
        Method to create an individual (from DeapOptimizer)
        """
        ps = [random.randint(self.hyperparam_space.evolvable_hyperparams[k].min_value,
                             self.hyperparam_space.evolvable_hyperparams[k].max_value)
              for k in self.hyperparam_space.evolvable_hyperparams.keys()]
        return pcls(ps)

    def params_to_individual(self, params: dict, pcls=None):
        """
        Convert a hyperparameter dictionary to an individual representation.

        This allows seeding the population with known good configurations
        (e.g., sklearn defaults or user-provided starting points).

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameter values (e.g., {'max_depth': 10, 'max_features': 0.2})
        pcls : class, optional
            The individual class to use (e.g., creator.Individual)

        Returns
        -------
        individual : list or pcls instance
            The individual representation
        """
        individual = []
        for key in self.hyperparam_space.evolvable_hyperparams.keys():
            hp = self.hyperparam_space.evolvable_hyperparams[key]

            if key in params and params[key] is not None:
                val = params[key]

                # Reverse the correct() operation to get integer representation
                if hp.hyperparam_type == 'int':
                    int_val = int(val)
                elif hp.hyperparam_type == 'float':
                    int_val = int(val * hp.scale)
                elif hp.hyperparam_type == 'nexp':
                    # val = 10^(-int_val), so int_val = -log10(val)
                    import math
                    int_val = int(-math.log10(val)) if val > 0 else hp.max_value
                elif hp.hyperparam_type == 'x10':
                    int_val = int(val / 10)
                elif hp.hyperparam_type == 'list':
                    # Find index of value in values_str
                    if val in hp.values_str:
                        int_val = hp.values_str.index(val)
                    else:
                        int_val = 0
                else:
                    int_val = int(val)

                # Clamp to valid range
                int_val = max(hp.min_value, min(hp.max_value, int_val))
            else:
                # Use middle of range as fallback for missing params
                int_val = (hp.min_value + hp.max_value) // 2

            individual.append(int_val)

        if pcls is not None:
            return pcls(individual)
        return individual

    def get_default_individual(self, estimator_class, pcls=None):
        """
        Create an individual representing sklearn's default hyperparameters.

        Parameters
        ----------
        estimator_class : class
            The sklearn estimator class to get defaults from
        pcls : class, optional
            The individual class to use

        Returns
        -------
        individual : list or pcls instance
            Individual representing sklearn defaults (clamped to search space)
        """
        # Get sklearn default params
        default_estimator = estimator_class()
        default_params = default_estimator.get_params()

        # Map special values to numeric equivalents
        mapped_params = {}
        for key in self.hyperparam_space.evolvable_hyperparams.keys():
            if key in default_params:
                val = default_params[key]

                # Handle special string values
                if val == 'sqrt':
                    # Use middle-low value as approximation
                    mapped_params[key] = 0.15
                elif val == 'log2':
                    mapped_params[key] = 0.10
                elif val == 'auto':
                    mapped_params[key] = 1.0
                elif val is None:
                    # For max_depth=None, use upper bound
                    hp = self.hyperparam_space.evolvable_hyperparams[key]
                    if hp.hyperparam_type in ['int', 'float']:
                        if hp.hyperparam_type == 'float':
                            mapped_params[key] = hp.max_value / hp.scale
                        else:
                            mapped_params[key] = hp.max_value
                else:
                    mapped_params[key] = val

        return self.params_to_individual(mapped_params, pcls)

class Individual:
    def __init__(self, genome, fitness=None):
        self.genome = genome
        self.fitness = fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def __repr__(self):
        return f"Individual(genome={self.genome}, fitness={self.fitness})"