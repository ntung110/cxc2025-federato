
import polars as pl
import torch
import pickle
import os


class EventPredictor:

    def __init__(self, event_path, retention_path, time_usage_path):

        # Save path
        self.event_path = event_path
        self.retention_path = retention_path
        self.time_usage_path = time_usage_path

        # Load models
        self.event_model = torch.load(self.event_path)

        with open(self.retention_path, 'rb') as file:
            self.retention_model = pickle.load(file)

        with open(self.time_usage_path, 'rb') as file:
            self.time_usage_model = pickle.load(file)


    def predict(self, df : pl.DataFrame):
        
        # Get predictions from models
        event_proba = recommend_next_action(df['user_sequence'])
        retention_y_pred_proba = self.retention_model.predict_proba(df.drop('returned_within_28_days_max'))[::, 1]
        time_usage_y_pred = self.time_usage_model.predict(df.drop('session_seconds_mean'))
        time_usage_y_pred = time_usage_y_pred / (time_usage_y_pred.quantile(0.75) - time_usage_y_pred.quantile(0.25))


        # Get next action
        next_action_labels = event_proba * time_usage_y_pred * retention_y_pred_proba
        return next_action_labels



if __name__ == '__main__':

    # Read and preprocess data
    file_path = os.path.expanduser('~/Desktop/data/preprocessed_data.parquet')
    df = pl.scan_parquet(file_path).limit(10)
    df = df.collect()
    print(df)

    # Get prediction
    event_predictor = EventPredictor(EVENT_MODEL_PATH, RETENTION_MODEL_PATH, TIME_USAGE_MODEL_PATH)
    next_action_labels = event_predictor.predict(df)
    print(next_action_labels)

