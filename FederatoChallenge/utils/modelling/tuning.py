from collections.abc import Callable
from optuna.samplers import TPESampler
import optuna

class Optimizer:

    """ Optimizer is class used for optimizing an Optuna objective"""

    def __init__(self, n_trials : int = 100, random_state : int = 42):

        # Initialize object fields
        self.n_trials = n_trials
        self.random_state = random_state

        optuna.logging.get_verbosity()
        optuna.logging.INFO
        # Create study
        self.sampler = TPESampler(seed = self.random_state)
        self.study = optuna.create_study(sampler = self.sampler, direction='minimize') # minimize logloss


    def optimize(self, objective: Callable) -> tuple:
        self.study.optimize(objective, n_trials=self.n_trials)  # Use the existing study
        return self.study.best_params, self.study.best_value


class HyperparameterObjective:

    """
    HyperparameterObjective is an class that makes a hyperparameter tuning objective,
    which tunes a model's hyperparameters using a validation set and an evaluation metric
    """

    def __init__(self, trainer, splitter):
        self.trainer = trainer
        self.splitter = splitter

    def make_objective(self, model, df, config : dict, metric, target, fixed_params=None) -> Callable:
        if fixed_params is None:
            fixed_params = {}
        def objective(trial):

            # Make sure that splitter is undeterminisitic to avoid overfitting
            self.splitter.random_state = None

            # Split data into training and validation features and targets
            df_train, df_val = self.splitter.data_split(df)
            X_train, y_train = self.splitter.X_y_split(df_train, target)
            X_val, y_val = self.splitter.X_y_split(df_val, target)
            def clean_data(X, y):
                mask = ~(y.is_null() | y.is_infinite())
                return X.filter(mask), y.filter(mask)

            X_train, y_train = clean_data(X_train, y_train)
            X_val, y_val = clean_data(X_val, y_val)
            # Suggest hyperparameter based on config
            model_params = fixed_params.copy()
            for name, attr in config.items():

                # Get variables from config
                suggest_vartype = f"suggest_{attr['type']}"
                suggest_params = [attr['params']['low'], attr['params']['high']]

                # Suggest corresponding value and save to model_params
                val = getattr(trial, suggest_vartype)(name, *suggest_params)
                model_params[name] = val

            # Train model
            fitted = self.trainer.fit_batch(X_train, y_train, model_params)
            y_pred = fitted.predict(X_val)
            return metric(y_val, y_pred)

        return objective
