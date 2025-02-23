import os
import math
import unittest
import polars as pl
from polars.testing import assert_frame_equal
from preprocessing import *


class TestBatchPreprocessData(unittest.TestCase):

    def add_one(self, df, config):
        return df.with_columns(pl.col(c) + 1 for c in df.columns)

    def setUp(self):
        self.init_df = pl.DataFrame({
            'x' : [i for i in range(1, 11)],
            'y' : [i for i in range(1, 11)]
        })
        self.output_path = './'
        self.config = {
                        'pipeline' : {
                            'input_path' : './',
                            'output_path' : './'
                        },

                        'preprocessing_pipeline' : {
                            'batch_preprocess_data' : {
                            'enabled' : True}
                        }
                    }
        
        self.chunk_size = 2

    def test_batch_preprocess_data(self):
        self.expected_df = pl.DataFrame({
            'x' : [i for i in range(2, 12)],
            'y' : [i for i in range(2, 12)]
        })
        self.res_df = batch_preprocess_data(self.init_df, self.add_one, self.config, self.chunk_size)
        assert_frame_equal(self.res_df, self.expected_df)
        
        # Count df files
        file_count = 0
        for file in os.listdir(self.output_path):
            if 'df' in file:
                file_count += 1

        self.assertEqual(file_count, math.ceil(self.init_df.shape[0] / self.chunk_size))


    def tearDown(self):
        for file in os.listdir(self.output_path):
            if 'df' in file:
                os.remove(file) 
    


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)