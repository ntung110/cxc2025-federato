from collections.abc import Callable
from optuna.samplers import TPESampler
import optuna

class Optimizer:

    """ Optimizer is class used for optimizing an Optuna objective"""

    def __init__(self, n_trials : int = 100, random_state : int = 42):

        # Initialize object fields
        self.n_trials = n_trials
        self.random_state = random_state

        # Create study
        self.sampler = TPESampler(seed = self.random_state)
        self.study = optuna.create_study(sampler = self.sampler)


    def optimize(self, objective : Callable) -> dict:
        study = optuna.create_study(objective, self.n_trials)
        return study.best_params


class HyperparameterObjective:

    """ 
    HyperparameterObjective is an class that makes a hyperparameter tuning objective,
    which tunes a model's hyperparameters using a validation set and an evaluation metric
    """

    def __init__(self, trainer, splitter):
        self.trainer = trainer
        self.splitter = splitter

    def make_objective(self, model, df, config : dict, metric) -> Callable:

        def objective(trial):

            # Make sure that splitter is undeterminisitic to avoid overfitting
            self.splitter.random_state = None

            # Split data into training and validation features and targets
            df_train, df_val = self.splitter.data_split(df)
            X_train, y_train = self.splitter.X_y_split(X_train, y_train)
            X_val, y_val = self.splitter.X_y_split(X_val, y_val)

            # Suggest hyperparameter based on config
            model_params = {}
            for name, attr in config.items():

                # Get variables from config
                suggest_vartype = f"suggest_{attr['type']}"
                suggest_params = attr['params']

                # Suggest corresponding value and save to model_params
                val = getattr(trial, suggest_vartype)(name, *attr['params'])
                model_params[name] = val

            # Train model 
            fitted = self.trainer.fit_batch(X_train, y_train, model_params) 
            y_pred = fitted.predict(X_val)
            return metric(y_pred, y_val)

        return objective