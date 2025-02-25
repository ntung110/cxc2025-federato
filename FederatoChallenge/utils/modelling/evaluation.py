import polars as pl
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, r2_score, 
								mean_squared_error, mean_absolute_error, accuracy_score, f1_score, median_absolute_error)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, List, Optional
import xgboost as xgb
import numpy as np
import pandas as pd



class Evaluation:
    """
    Evaluation is the parent class of ClassificationEvaluation and RegressionEvaluation.
    """
    def __init__(self, model):
        """
        Initialize with a model
        """
        self.model = model
       
    def plot_xgb_feature_importance(self):
        ax = xgb.plot_importance(self.model, show_values = False)
        plt.show()
        return ax
    
    
    def shuffling_feature_importance(self, X, y):
        y_array = y.to_numpy() if isinstance(y, pl.Series) else y

        # Calculate the baseline performance using RMSE
        baseline = np.sqrt(mean_squared_error(y_array, self.model.predict(X)))

        importances = {}

        # Iterate over each feature to shuffle
        for col in X.columns:
            # Create a copy of X_test (Polars DataFrames are immutable, so we use with_columns)
            X_shuffled = X.clone()

            # Shuffle the column using numpy's permutation and replace it in the DataFrame
            shuffled_values = np.random.permutation(X_shuffled[col].to_numpy())
            X_shuffled = X_shuffled.with_columns(pl.Series(col, shuffled_values))

            # Measure performance after shuffling
            shuffled_rmse = np.sqrt(mean_squared_error(y_array, self.model.predict(X_shuffled)))

            # Calculate the increase in RMSE (higher = more important)
            importances[col] = shuffled_rmse - baseline

            # Convert the importances dictionary to a Polars DataFrame
        importance_df = pl.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": list(importances.values())
        })

        importance_df = importance_df.sort("Importance", descending=True)

        plt.figure(figsize=(10, 6))
        plt.bar(importance_df["Feature"], importance_df["Importance"], color='skyblue')
        plt.xlabel("Features")
        plt.ylabel("Importance (Increase in RMSE)")
        plt.title("Shuffling Feature Importance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return importance_df



class ClassificationEvaluation(Evaluation):
    """
    ClassificationEvaluation is an evaluation framework for classification models.
    """
    def __init__(self, model):
        self.model = model
   
    def get_classification_metrics(self, X, y):
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)
       
        clf_metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'F1': f1_score(y, y_pred),
            'AUC': roc_auc_score(y, y_pred_proba),
            'Confusion Matrix': confusion_matrix(y, y_pred)
        }

        for name, score in clf_metrics.items():
            print(f'{name}:\n{score}')
       
        # return clf_metrics
   
    def plot_roc_curve(self, X, y):
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)
        plt.plot(fpr, tpr, label="AUC=" + str(round(auc, 3)))
        plt.legend(loc=4)
        plt.show()

    def plot_learning_curve(self):
        results = self.model.evals_result()
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
   
        plt.figure(figsize=(6, 6))
        plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
        plt.xlabel("Epochs")
        plt.ylabel("Log Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()


class RegressionEvaluation(Evaluation):
    """
    RegressionEvaluation is an evaluation framework for regression models
    which outputs the relevant metrics and figures for analysis.
    """
    def __init__(self, model):
        super().__init__(model)
   
    def get_regression_metrics(self, X, y):
        y_pred = self.model.predict(X)
        reg_metrics = {
            'R2': r2_score(y, y_pred),
            'MSE': mean_squared_error(y, y_pred),
            'MAD': median_absolute_error(y, y_pred),
        }

        for name, score in reg_metrics.items():
            print(f'{name}:\n{score}')
        # return reg_metrics

    def plot_learning_curve(self):
        results = self.model.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
   
        plt.figure(figsize=(6, 6))
        plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
        plt.plot(x_axis, results['validation_1']['rmse'], label='Validation')
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()