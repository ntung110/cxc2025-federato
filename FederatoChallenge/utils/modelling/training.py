
from xgboost import XGBRegressor, XGBClassifier


class Trainer:
    def __init__(self, model_cls, dataloader):
        self.model_cls = model_cls
        self.dataloader = dataloader

    def fit(self, X, y, params):

        # Initialize and train model
        self.model = self.model_cls(**params)
        self.model.fit(X, y)
        return self.model

    def fit_batch(self, X, y, params):
       
        # Initialize model
        not_trained = True
        self.model = self.model_cls(**params)

        # Initialize dataloader and train by batch
        self.dataloader.fit(X, y)
        for X_batch, y_batch in self.dataloader:
            if not_trained:
                self.model.fit(X_batch, y_batch)
                not_trained = False
            else:
                self.model.fit(X_batch, y_batch, xgb_model = self.model.get_booster())
        return self.model
