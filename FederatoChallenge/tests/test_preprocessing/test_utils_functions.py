import unittest
import polars as pl
from polars.testing import assert_frame_equal
from utils.preprocessing import *


class TestDropColumns(unittest.TestCase):

    def setUp(self):
        self.init_df = pl.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NY", "LA", "SF"]
        })

        self.config = {'preprocessing_pipeline' : {
                        'drop_columns' : {
                        'enabled' : True,
                        'params' : {
                            'columns' : ['age', 'city']
                        }
                    }}}



    def test_drop_columns(self):
        self.expected_df =  pl.DataFrame({
            "name": ["Alice", "Bob", "Charlie"]})
        
        self.res_df = drop_columns(self.init_df, self.config)
        assert_frame_equal(self.res_df, self.expected_df)
        

class TestFillMissingValues(unittest.TestCase):

    def setUp(self):
        '''Set up initial DataFrame and configuration for testing'''
        self.init_df = pl.DataFrame({
            "A": [10, None, 30, None, 50], 
            "B": [None, "cat", "dog", None, "mouse"], 
            "C": [100, 200, None, 400, 500]
        })

        self.config = {
            "preprocessing_pipeline": {
                "fill_missing_values": {
                    "enabled": True,
                    "params": {}
                }
            }
        }

    def test_mean_fill(self):
        '''Test filling missing values using mean strategy'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "mean"}
        result_df = fill_missing_values(self.init_df, self.config)

        expected_A_mean = self.init_df["A"].mean()
        expected_C_mean = self.init_df["C"].mean()

        self.assertEqual(result_df["A"][1], expected_A_mean)
        self.assertEqual(result_df["A"][3], expected_A_mean)
        self.assertEqual(result_df["C"][2], expected_C_mean)

    def test_median_fill(self):
        '''Test filling missing values using median strategy'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "median"}
        result_df = fill_missing_values(self.init_df, self.config)

        expected_A_median = self.init_df["A"].median()
        expected_C_median = self.init_df["C"].median()

        self.assertEqual(result_df["A"][1], expected_A_median)
        self.assertEqual(result_df["A"][3], expected_A_median)
        self.assertEqual(result_df["C"][2], expected_C_median)

    def test_mode_fill(self):
        '''Test filling missing values using mode strategy'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "mode"}
        result_df = fill_missing_values(self.init_df, self.config)

        expected_B_mode = self.init_df["B"].mode()[0]

        self.assertEqual(result_df["B"][0], expected_B_mode)
        self.assertEqual(result_df["B"][3], expected_B_mode)

    def test_constant_fill(self):
        '''Test filling missing values using a constant value'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {
            "strategy": "constant",
            "constant_value": "missing"
        }
        result_df = fill_missing_values(self.init_df, self.config)

        self.assertEqual(result_df["A"][1], "missing")
        self.assertEqual(result_df["B"][0], "missing")
        self.assertEqual(result_df["C"][2], "missing")

    def test_forward_fill(self):
        '''Test filling missing values using forward fill strategy'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "forward"}
        result_df = fill_missing_values(self.init_df, self.config)

        self.assertEqual(result_df["A"][1], 10)  
        self.assertEqual(result_df["A"][3], 30)  
        self.assertEqual(result_df["B"][0], None)  
        self.assertEqual(result_df["B"][3], "dog")

    def test_backward_fill(self):
        '''Test filling missing values using backward fill strategy'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "backward"}
        result_df = fill_missing_values(self.init_df, self.config)

        self.assertEqual(result_df["A"][1], 30)  
        self.assertEqual(result_df["A"][3], 50)  
        self.assertEqual(result_df["B"][0], "cat")
        self.assertEqual(result_df["B"][3], "mouse")

    def test_min_fill(self):
        '''Test filling missing values using min strategy'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "min"}
        result_df = fill_missing_values(self.init_df, self.config)

        self.assertEqual(result_df["A"][1], 10)  
        self.assertEqual(result_df["C"][2], 100)  
        
    def test_max_fill(self):
        '''Test filling missing values using max strategy'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "max"}
        result_df = fill_missing_values(self.init_df, self.config)

        self.assertEqual(result_df["A"][1], 50)  
        self.assertEqual(result_df["C"][2], 500)  
        
    def test_zero_fill(self):
        '''Test filling missing values using zero strategy'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "zero"}
        result_df = fill_missing_values(self.init_df, self.config)

        self.assertEqual(result_df["A"][1], 0)
        self.assertEqual(result_df["C"][2], 0)

    def test_disabled_fill(self):
        '''Test when missing value filling is disabled (should return original DataFrame)'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["enabled"] = False
        result_df = fill_missing_values(self.init_df, self.config)

        self.assertTrue(result_df.equals(self.init_df))

    def test_empty_dataframe(self):
        '''Test filling missing values on an empty DataFrame'''
        empty_df = pl.DataFrame({})
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "mean"}
        result_df = fill_missing_values(empty_df, self.config)

        self.assertEqual(result_df.shape, (0, 0)) 

    def test_missing_column(self):
        '''Test when the specified column for missing value filling does not exist'''
        df_missing_col = pl.DataFrame({"D": [None, None, None]})  # Only missing values
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "mean"}
        result_df = fill_missing_values(df_missing_col, self.config)

        self.assertTrue(result_df["D"].is_null().sum() == 3) 

    def test_invalid_strategy(self):
        '''Test when an invalid strategy is provided'''
        self.config["preprocessing_pipeline"]["fill_missing_values"]["params"] = {"strategy": "invalid_strategy"}

        with self.assertRaises(ValueError):
            fill_missing_values(self.init_df, self.config)



class TestReplaceWithNull(unittest.TestCase):

    def setUp(self):
        self.init_df = pl.DataFrame({
            "name": ["Alice", "Bob", "Alice"],
            "age": [25, 30, 35],
            "city": ["NY", "LA", "SF"]
        })

        self.config = {'preprocessing_pipeline' : {
                        'replace_with_null' : {
                            'enabled' : True,
                            'params' : {
                                'null_vals' : ['Alice', 'NY']
                            }
                        }}}
        
    def test_replace_with_null(self):
        self.expected_df = pl.DataFrame({
            "name": [None, "Bob", None],
            "age": [25, 30 , 35],
            "city": [None, "LA", "SF"]
        })
        self.res_df = replace_with_null(self.init_df, self.config)
        assert_frame_equal(self.res_df, self.expected_df)
        
        
    
class TestEncodeCategorical(unittest.TestCase):
    
    def setUp(self):
        self.init_df = pl.DataFrame({
            "A": ["red", "blue", "green", "blue"],
            "B": ["apple", "banana", "apple", "banana"],
            "C": [1, 2, 3, 4] 
        })
        
        self.config = {
            "Preprocessing_pipeline": {
                "encode_categorical": {
                    "enabled": True,
                    "params": {}
                }
            }
        }
        
    def test_one_hot_encoding(self):
        '''Test one-hot encoding without dropping first category'''
        self.config["Preprocessing_pipeline"]["encode_categorical"]["params"] = {
            "method" : "one_hot",
            "drop_first": False
        }

        expected_df = pl.DataFrame({
        "A_blue": [0, 1, 0, 1],
        "A_green": [0, 0, 1, 0],
        "A_red": [1, 0, 0, 0],
        "B_apple": [1, 0, 1, 0],
        "B_banana": [0, 1, 0, 1],
        "C": [1, 2, 3, 4]
        })
        
        result_df = encode_categorical(self.init_df, self.config)
        self.assertTrue(result_df.equals(expected_df))
            
    def test_one_hot_encoding_drop_first(self):
        '''Test one-hot encoding with drop_first=True'''
        self.config["Preprocessing_pipeline"]["encode_categorical"]["params"] = {
            "method": "one_hot",
            "drop_first": True
        }
        
        expected_df = pl.DataFrame({
            "A_blue": [0, 1, 0, 1],  
            "A_green": [0, 0, 1, 0],
            "B_banana": [0, 1, 0, 1],  
            "C": [1, 2, 3, 4]
        })

        result_df = encode_categorical(self.init_df, self.config)
        self.assertTrue(result_df.equals(expected_df))

    def test_label_encoding(self):
        '''Test label encoding'''
        self.config["Preprocessing_pipeline"]["encode_categorical"]["params"] = {
            "method": "label"
        }
        
        result_df = encode_categorical(self.init_df, self.config)

        self.assertEqual(result_df["A"].dtype, pl.Int64)
        self.assertEqual(result_df["B"].dtype, pl.Int64)
        self.assertEqual(result_df["C"].dtype, pl.Int64)  

        self.assertEqual(len(set(result_df["A"].to_list())), 3) 
        self.assertEqual(len(set(result_df["B"].to_list())), 2) 

    def test_ordinal_encoding(self):
        '''Test ordinal encoding'''
        self.config["Preprocessing_pipeline"]["encode_categorical"]["params"] = {
            "method": "ordinal"
        }
        
        result_df = encode_categorical(self.init_df, self.config)

        self.assertEqual(result_df["A"].dtype, pl.Int64)
        self.assertEqual(result_df["B"].dtype, pl.Int64)
        self.assertEqual(result_df["C"].dtype, pl.Int64)

    def test_disabled_encoding(self):
        '''Test when encoding is disabled (should return original DataFrame)'''
        self.config["Preprocessing_pipeline"]["encode_categorical"]["enabled"] = False

        result_df = encode_categorical(self.init_df, self.config)
        self.assertTrue(result_df.equals(self.init_df))

    def test_empty_dataframe(self):
        '''Test encoding with an empty DataFrame'''
        empty_df = pl.DataFrame({})
        self.config["Preprocessing_pipeline"]["encode_categorical"]["params"] = {
            "method": "one_hot"
        }

        result_df = encode_categorical(empty_df, self.config)
        self.assertEqual(result_df.shape, (0, 0)) 

    def test_unseen_categories(self):
        '''Test label encoding with unseen categories in test data'''
        df_train = pl.DataFrame({"A": ["apple", "banana", "cherry"]})
        df_test = pl.DataFrame({"A": ["banana", "apple", "grape"]}) 

        self.config["Preprocessing_pipeline"]["encode_categorical"]["params"] = {
            "method": "label"
        }

        encoded_train = encode_categorical(df_train, self.config)
        encoded_test = encode_categorical(df_test, self.config)

        self.assertEqual(encoded_train["A"].dtype, pl.Int64)
        self.assertEqual(encoded_test["A"].dtype, pl.Int64)
        self.assertEqual(len(set(encoded_train["A"].to_list())), 3)
        self.assertIn(len(set(encoded_test["A"].to_list())), {2, 3})
        
        
class TestScaleFeatures(unittest.TestCase):

    def setUp(self):
        '''Set up initial DataFrame and configuration for testing'''
        self.init_df = pl.DataFrame({
            "A": [10, 20, 30, 40, 50],
            "B": [5, 15, 25, 35, 45],
            "C": [100, 200, 300, 400, 500]
        })

        self.config = {
            "preprocessing_pipeline": {
                "scale_features": {
                    "enabled": True,
                    "params": {}
                }
            }
        }

    def test_standard_scaling(self):
        '''Test standard (Z-score) scaling'''
        self.config["preprocessing_pipeline"]["scale_features"]["params"] = {
            "standard": ["A", "B"]
        }

        result_df = scale_features(self.init_df, self.config)

        expected_A = (self.init_df["A"] - self.init_df["A"].mean()) / self.init_df["A"].std()
        expected_B = (self.init_df["B"] - self.init_df["B"].mean()) / self.init_df["B"].std()

        self.assertAlmostEqual(result_df["A"].mean(), 0, places=5)
        self.assertAlmostEqual(result_df["A"].std(), 1, places=5)
        self.assertAlmostEqual(result_df["B"].mean(), 0, places=5)
        self.assertAlmostEqual(result_df["B"].std(), 1, places=5)

        self.assertTrue(result_df["A"].to_list() == expected_A.to_list())
        self.assertTrue(result_df["B"].to_list() == expected_B.to_list())

    def test_minmax_scaling(self):
        '''Test Min-Max scaling'''
        self.config["preprocessing_pipeline"]["scale_features"]["params"] = {
            "minmax": ["A", "B"]
        }

        result_df = scale_features(self.init_df, self.config)

        expected_A = (self.init_df["A"] - self.init_df["A"].min()) / (self.init_df["A"].max() - self.init_df["A"].min())
        expected_B = (self.init_df["B"] - self.init_df["B"].min()) / (self.init_df["B"].max() - self.init_df["B"].min())

        self.assertTrue(all(0 <= x <= 1 for x in result_df["A"].to_list()))
        self.assertTrue(all(0 <= x <= 1 for x in result_df["B"].to_list()))

        self.assertTrue(result_df["A"].to_list() == expected_A.to_list())
        self.assertTrue(result_df["B"].to_list() == expected_B.to_list())

    def test_robust_scaling(self):
        '''Test Robust scaling (based on IQR)'''
        self.config["preprocessing_pipeline"]["scale_features"]["params"] = {
            "robust": ["A", "B"]
        }

        result_df = scale_features(self.init_df, self.config)

        median_A = self.init_df["A"].median()
        Q1_A = self.init_df["A"].quantile(0.25)
        Q3_A = self.init_df["A"].quantile(0.75)

        expected_A = (self.init_df["A"] - median_A) / (Q3_A - Q1_A)

        median_B = self.init_df["B"].median()
        Q1_B = self.init_df["B"].quantile(0.25)
        Q3_B = self.init_df["B"].quantile(0.75)

        expected_B = (self.init_df["B"] - median_B) / (Q3_B - Q1_B)

        self.assertTrue(result_df["A"].to_list() == expected_A.to_list())
        self.assertTrue(result_df["B"].to_list() == expected_B.to_list())

    def test_scaling_disabled(self):
        '''Test if scaling is disabled, the DataFrame should remain unchanged'''
        self.config["preprocessing_pipeline"]["scale_features"]["enabled"] = False

        result_df = scale_features(self.init_df, self.config)
        self.assertTrue(result_df.equals(self.init_df))

    def test_empty_dataframe(self):
        '''Test scaling with an empty DataFrame'''
        empty_df = pl.DataFrame({})
        self.config["preprocessing_pipeline"]["scale_features"]["params"] = {
            "standard": ["A"]
        }

        result_df = scale_features(empty_df, self.config)
        self.assertEqual(result_df.shape, (0, 0))  

    def test_missing_column(self):
        '''Test when the column to be scaled is missing in DataFrame'''
        self.config["preprocessing_pipeline"]["scale_features"]["params"] = {
            "standard": ["D"] 
        }

        result_df = scale_features(self.init_df, self.config)
        self.assertTrue(result_df.equals(self.init_df)) 

    def test_mixed_scaling_methods(self):
        '''Test applying different scaling methods to different columns'''
        self.config["preprocessing_pipeline"]["scale_features"]["params"] = {
            "standard": ["A"],
            "minmax": ["B"],
            "robust": ["C"]
        }

        result_df = scale_features(self.init_df, self.config)

        expected_A = (self.init_df["A"] - self.init_df["A"].mean()) / self.init_df["A"].std()
        expected_B = (self.init_df["B"] - self.init_df["B"].min()) / (self.init_df["B"].max() - self.init_df["B"].min())

        median_C = self.init_df["C"].median()
        Q1_C = self.init_df["C"].quantile(0.25)
        Q3_C = self.init_df["C"].quantile(0.75)
        expected_C = (self.init_df["C"] - median_C) / (Q3_C - Q1_C)

        self.assertTrue(result_df["A"].to_list() == expected_A.to_list())
        self.assertTrue(result_df["B"].to_list() == expected_B.to_list())
        self.assertTrue(result_df["C"].to_list() == expected_C.to_list())

          
          
class TestHandleOutliers(unittest.TestCase):

    def setUp(self):
        '''Set up initial DataFrame and configuration for testing'''
        self.init_df = pl.DataFrame({
            "A": [10, 200, 15, 3000, 25],  
            "B": [5, 15, 25, 35, 45],      
            "C": [100, 200, 300, 400, 500]  
        })

        self.config = {
            "preprocessing_pipeline": {
                "handle_outliers": {
                    "enabled": True,
                    "params": {}
                }
            }
        }

    def test_z_score_outlier_removal(self):
        '''Test Z-score based outlier handling'''
        self.config["preprocessing_pipeline"]["handle_outliers"]["params"] = {
            "threshold": 2,
            "method": {"z-score": ["A", "B"]}
        }

        result_df = handle_outliers(self.init_df, self.config)

        mean_A = self.init_df["A"].mean()
        std_A = self.init_df["A"].std()
        threshold = 2
        expected_A = [(x if abs((x - mean_A) / std_A) <= threshold else mean_A) for x in self.init_df["A"].to_list()]

        self.assertTrue(result_df["A"].to_list() == expected_A)
        self.assertTrue(result_df["B"].to_list() == self.init_df["B"].to_list())

    def test_iqr_outlier_removal(self):
        '''Test IQR-based outlier handling (filters outliers)'''
        self.config["preprocessing_pipeline"]["handle_outliers"]["params"] = {
            "method": {"iqr": ["A"]}
        }

        result_df = handle_outliers(self.init_df, self.config)

        Q1 = self.init_df["A"].quantile(0.25)
        Q3 = self.init_df["A"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        expected_A = [x for x in self.init_df["A"].to_list() if lower_bound <= x <= upper_bound]
        self.assertTrue(result_df["A"].to_list() == expected_A)

    def test_winsorization(self):
        '''Test Winsorization-based outlier handling'''
        self.config["preprocessing_pipeline"]["handle_outliers"]["params"] = {
            "method": {"winsorization": ["A"]}
        }

        result_df = handle_outliers(self.init_df, self.config)

        lower_bound = self.init_df["A"].quantile(0.05)
        upper_bound = self.init_df["A"].quantile(0.95)

        expected_A = [min(max(x, lower_bound), upper_bound) for x in self.init_df["A"].to_list()]
        self.assertTrue(result_df["A"].to_list() == expected_A)

    def test_disabled_outlier_handling(self):
        '''Test when outlier handling is disabled (should return original DataFrame)'''
        self.config["preprocessing_pipeline"]["handle_outliers"]["enabled"] = False

        result_df = handle_outliers(self.init_df, self.config)
        self.assertTrue(result_df.equals(self.init_df))

    def test_empty_dataframe(self):
        '''Test handling outliers on an empty DataFrame'''
        empty_df = pl.DataFrame({})
        self.config["preprocessing_pipeline"]["handle_outliers"]["params"] = {
            "method": {"z-score": ["A"]}
        }

        result_df = handle_outliers(empty_df, self.config)
        self.assertEqual(result_df.shape, (0, 0)) 

    def test_missing_column(self):
        '''Test when the specified column for outlier removal is missing'''
        self.config["preprocessing_pipeline"]["handle_outliers"]["params"] = {
            "method": {"z-score": ["D"]}  
        }

        result_df = handle_outliers(self.init_df, self.config)
        self.assertTrue(result_df.equals(self.init_df))  

    def test_mixed_methods(self):
        '''Test applying different outlier methods to different columns'''
        self.config["preprocessing_pipeline"]["handle_outliers"]["params"] = {
            "threshold": 3,
            "method": {
                "z-score": ["A"],
                "iqr": ["B"],
                "winsorization": ["C"]
            }
        }

        result_df = handle_outliers(self.init_df, self.config)

        mean_A = self.init_df["A"].mean()
        std_A = self.init_df["A"].std()
        expected_A = [(x if abs((x - mean_A) / std_A) <= 3 else mean_A) for x in self.init_df["A"].to_list()]

        Q1_B = self.init_df["B"].quantile(0.25)
        Q3_B = self.init_df["B"].quantile(0.75)
        IQR_B = Q3_B - Q1_B
        lower_B, upper_B = Q1_B - 1.5 * IQR_B, Q3_B + 1.5 * IQR_B
        expected_B = [x for x in self.init_df["B"].to_list() if lower_B <= x <= upper_B]

        lower_C = self.init_df["C"].quantile(0.05)
        upper_C = self.init_df["C"].quantile(0.95)
        expected_C = [min(max(x, lower_C), upper_C) for x in self.init_df["C"].to_list()]

        self.assertTrue(result_df["A"].to_list() == expected_A)
        self.assertTrue(result_df["B"].to_list() == expected_B)
        self.assertTrue(result_df["C"].to_list() == expected_C)

    def test_handling_of_non_numeric_columns(self):
        '''Ensure non-numeric columns are ignored'''
        df_with_strings = self.init_df.with_columns(pl.Series("D", ["a", "b", "c", "d", "e"]))

        self.config["preprocessing_pipeline"]["handle_outliers"]["params"] = {
            "method": {"z-score": ["A", "D"]} 
        }

        result_df = handle_outliers(df_with_strings, self.config)

        self.assertTrue(result_df["D"].to_list() == df_with_strings["D"].to_list())

    def test_invalid_threshold_value(self):
        '''Test invalid threshold value (should default to 3)'''
        self.config["preprocessing_pipeline"]["handle_outliers"]["params"] = {
            "threshold": "invalid",
            "method": {"z-score": ["A"]}
        }

        result_df = handle_outliers(self.init_df, self.config)

        mean_A = self.init_df["A"].mean()
        std_A = self.init_df["A"].std()
        expected_A = [(x if abs((x - mean_A) / std_A) <= 3 else mean_A) for x in self.init_df["A"].to_list()]

        self.assertTrue(result_df["A"].to_list() == expected_A)          
          
            
class TestLowercaseAllValues(unittest.TestCase):

    def setUp(self):
        self.init_df = pl.DataFrame({
            "name": ["Alice", "Bob", "Alice"],
            "age": ['EIGHTEEN', 'eightEEN', 'eIghTeen'],
            "city": ["NY", "LA", "SF"]
        })

        self.config = {'preprocessing_pipeline' : {
                        'lowercase_all_values' : {
                        'enabled' : True
                    }}}

    def test_lowercase_all_values(self):
        self.expected_df = pl.DataFrame({
            "name": ["alice", "bob", "alice"],
            "age": ['eighteen', 'eighteen', 'eighteen'],
            "city": ["ny", "la", "sf"]
        })
        self.res_df = lowercase_all_values(self.init_df, self.config)
        assert_frame_equal(self.res_df, self.expected_df)



class TestCustomEnforceTypes(unittest.TestCase):

    def setUp(self):
        self.init_df = pl.DataFrame({
            "name": ["Alice", "Bob"],
            "age": ["25", "30"],
            "height": ["5.5", "6.0"],
            "birthday" : ['01-01-2024', '01-01-2025']
        })

        self.config = {
            "preprocessing_pipeline": {
                "custom_enforce_types": {
                    "enabled": True,
                    "params": {
                        "per_column": {
                            "int": ["age"],
                            "float" : ["height"],
                            "string": ["name"],
                            "datetime" : ["birthday"]
                        }
                    }
                }
            },
            "datatype": {
                "int" : "Int64",
                "float": "Float64",
                "string": "String",
                "datetime" : "Datetime"
            }
        }

    def test_custom_enforce_types(self):
        result = custom_enforce_types(self.init_df, self.config)

        self.assertEqual(result.schema["age"], pl.Int64)
        self.assertEqual(result.schema["height"], pl.Float64)
        self.assertEqual(result.schema["name"], pl.String)
        self.assertEqual(result.schema["birthday"], pl.Datetime)



if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)