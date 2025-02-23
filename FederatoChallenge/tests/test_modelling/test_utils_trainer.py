import unittest
import polars as pl
from xgboost import XGBRegressor, XGBClassifier
from utils.modelling import Dataloader, Trainer

class TestTrainer(unittest.TestCase):
    
    def setUp(self):
        self.df = pl.DataFrame({"A": range(100), "B": range(100, 200)})
        self.X, self.y = 
        self.target = pl.Series("target", range(100))
        self.dataloader = Dataloader(batch_size=10)
        
    def test_fit_regressor(self):
        trainer = Trainer(XGBRegressor, self.dataloader)
        model = trainer.fit(self.df, self.target, params={"objective": "reg:squarederror"})
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
    
    def test_fit_classifier(self):
        trainer = Trainer(XGBClassifier, self.dataloader)
        model = trainer.fit(self.df, self.target, params={"objective": "binary:logistic"})
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
    
    def test_fit_batch(self):
        trainer = Trainer(XGBRegressor, self.dataloader)
        model = trainer.fit_batch(self.df, self.target, params={"objective": "reg:squarederror"})
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
    
    def test_invalid_model(self):
        with self.assertRaises(TypeError):
            Trainer(None, self.dataloader).fit(self.df, self.target, params={})
    
    def test_dataloader_integration(self):
        trainer = Trainer(XGBRegressor, self.dataloader)
        self.dataloader.fit(self.df, self.target)
        for X_batch, y_batch in self.dataloader:
            self.assertEqual(len(X_batch), 10)  # Ensure batch size is correct


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)