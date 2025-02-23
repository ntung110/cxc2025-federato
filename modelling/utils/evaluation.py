import polars as pl
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, r2_score, 
								mean_squared_error, mean_absolute_error, accuracy_score, f1_score, median_absolute_error)
from sklearn.model_selection import train_test_split, learn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, List, Optional
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from modelling import Dataloader


class Evaluation:
	"""
	Evaluation is the parent class of ClassificationEvaluation and RegressionEvaluation.
	"""
	def __init__(self, model: Union[xgb.XGBClassifier, xgb.XGBRegressor], dataloader : Dataloader):
		"""
		Initialize with a model
		"""
		self.model = model
		self.dataloader = dataloader
		

	def plot_feature_importance(self, X: pl.DataFrame, y: pl.Series):
		ax = xgb.plot_importance(self.model)
		plt.show()
		return ax
	
	def plot_learning_curve(self):
		results = self.model.evals_result()
		epochs = len(results['validation_0']['logloss'])
		x_axis = range(0, epochs)

		fig, ax = plt.figure(figsize=(6, 6))
		plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
		plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
		plt.xlabel("Epochs")
		plt.ylabel("Log Loss")
		plt.title("Learning Curve")
		plt.legend()
		plt.show()
		return ax
		

class ClassificationEvaluation(Evaluation): 
	"""
	ClassificationEvaluation is an evaluation framework for classification models

	"""

	def __init__(self, model):
		super().__init__(model)


	def get_classification_metrics(self, X : pl.DataFrame, y : pl.Series):

		# Calculate metrics
		y_pred_proba = self.model.predict_proba(X)[::, 1]
		y_pred = self.model.predict(X)
		clf_metrics = {
			'Accuracy' : accuracy_score(y, y_pred),
			'F1' : f1_score(y, y_pred),
			'AUC' : roc_auc_score(y, y_pred_proba),
			'Confusion Matrix' : confusion_matrix(y, y_pred)
		}

		# Print classification metrics
		for name, score in clf_metrics.items():
			print(name, ':\n')
			print(score)

		return clf_metrics
    
    
	def plot_roc_curve(self, X : pl.DataFrame, y : pl.Series):
		y_pred_proba = self.model.predict_proba(X)[::, 1]
		fpr, tpr, _ = roc_curve(y, y_pred_proba)
		auc = roc_auc_score(y, y_pred_proba)
		plt.plot(fpr,tpr,label="AUC="+str(round(auc, 3)))
		plt.legend(loc=4)
		plt.show()
		

class RegressionEvaluation(Evaluation):

	""" 
	RegressionEvaluation is an evaluation framework for regression models 
	which outputs the relevant metrics and figures for analysis
	"""

	def __init__(self, model):
		super().__init__(model)

	def get_regression_metrics(self, X, y):
		y_pred = self.model.predict(X)
		reg_metrics = {
			'R2' : r2_score(y, y_pred),
			'MSE' : mean_squared_error(y, y_pred),
			'MAD' : median_absolute_error(y, y_pred), 
		}

		# Print classification metrics
		for name, score in reg_metrics.items():
			print(name, ':\n')
			print(score)

		return reg_metrics
		