import unittest
import polars as pl
from datetime import datetime
from utils.modelling import Dataloader, TrainValidationSplit, TimeSplit

class TestDataloader(unittest.TestCase):
    
    def setUp(self):
        self.df = pl.DataFrame({"A": range(10), "B": range(10, 20)})
        self.target = pl.Series("target", range(10))
        self.batch_size = 3
        self.loader = Dataloader(batch_size=self.batch_size)
        self.loader.fit(self.df, self.target)
    
    def test_batch_loading(self):
        batches = list(self.loader)
        self.assertEqual(len(batches), 4)  # 3 full batches + 1 smaller batch
        self.assertEqual(len(batches[-1][0]), 1)  # Last batch should have 1 row

    def test_iteration_reset(self):
        list(self.loader)  # Consume all batches
        with self.assertRaises(StopIteration):
            next(self.loader)

class TestTrainValidationSplit(unittest.TestCase):
    
    def setUp(self):
        self.df = pl.DataFrame({"A": range(100), "B": range(100, 200), "target": range(100)})
        self.splitter = TrainValidationSplit(val_ratio=0.2)
    
    def test_X_y_split(self):
        X, y = self.splitter.X_y_split(self.df, "target")
        self.assertNotIn("target", X.columns)
        self.assertEqual(y.name, "target")
    
    def test_data_split(self):
        df_train, df_val = self.splitter.data_split(self.df)
        self.assertEqual(len(df_val), 20)
        self.assertEqual(len(df_train), 80)

class TestTimeSplit(unittest.TestCase):
    
    def setUp(self):
        self.df = pl.DataFrame({
            "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
            "value": [10, 20, 30, 40],
            "target": [0, 1, 0, 1]
        }).with_columns(pl.col("date").str.to_date())
        self.splitter = TimeSplit("2023-02-15")
    
    def test_X_y_split(self):
        X, y = self.splitter.X_y_split(self.df, "target")
        self.assertNotIn("target", X.columns)
        self.assertEqual(y.name, "target")
    
    def test_data_split(self):
        df_test, df_leftover = self.splitter.data_split(self.df, "date")
        self.assertEqual(len(df_test), 2)  # Dates after cutoff
        self.assertEqual(len(df_leftover), 2)  # Dates before or on cutoff


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
    