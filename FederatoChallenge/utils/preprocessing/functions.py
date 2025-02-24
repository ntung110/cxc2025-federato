import polars as pl
import logging
import ast
import os

from pathlib import Path
from scipy.stats import zscore
from pathlib import Path
from collections.abc import Callable


def drop_columns(df: pl.DataFrame, config: dict, run : int = 0) -> pl.DataFrame:
    if config["preprocessing_pipeline"]["drop_columns"]["enabled"]:
        params = config["preprocessing_pipeline"]["drop_columns"]["params"]
        columns_to_drop = params[f"columns_{run}"]
        df = df.drop(columns_to_drop)
    return df


def replace_with_null(df : pl.DataFrame, config : dict) -> pl.DataFrame:

    if config['preprocessing_pipeline']['replace_with_null']['enabled']:
    
        # Get params and replace with null
        params = config['preprocessing_pipeline']['replace_with_null']['params']
        null_vals = params['null_vals']
        return df.with_columns(
                pl.when(pl.col(c).is_in(null_vals))
                .then(None)
                .otherwise(pl.col(c)).name.keep()
            for c in df.select(pl.col(pl.String)).columns
        )


def encode_categorical(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    if not config["preprocessing_pipeline"]["encode_categorical"]["enabled"]:
        return df

    if df.is_empty():
        return df

    params = config["preprocessing_pipeline"]["encode_categorical"]["params"]
    method = params.get("method", "one_hot")
    drop_first = params.get("drop_first", False)

    categorical_cols = config["preprocessing_pipeline"]["encode_categorical"]["params"]["include_cols"]
    if not categorical_cols: 
        return df

    if method == "one_hot":
        df = df.to_dummies(columns=categorical_cols, drop_first=drop_first)

    elif method in ["label", "ordinal"]:
        col_mappings = []
        for col in categorical_cols:
            unique_vals = sorted(df[col].unique().to_list()) if method == "ordinal" else df[col].unique().to_list()
            mapping_dict = {val: i for i, val in enumerate(unique_vals)}

            # Correct application of mapping using `replace`
            mapped_series = df[col].replace(mapping_dict).cast(pl.Int64).alias(col)
            col_mappings.append(mapped_series)

        df = df.with_columns(col_mappings)

    return df



def scale_features(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    if config["preprocessing_pipeline"]["scale_features"]["enabled"]:
        params = config["preprocessing_pipeline"]["scale_features"]["params"]  

        for method, columns in params.items():
            for col in columns:
                if col in df.columns and columns:
                    if method == "standard":
                        mean = df[col].mean()
                        std = df[col].std()
                        df = df.with_columns(
                            ((pl.col(col) - mean) / std).alias(col)
                        )

                    elif method == "minmax":
                        min_val = df[col].min()
                        max_val = df[col].max()
                        df = df.with_columns(
                            ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
                        )

                    elif method == "robust":
                        median = df[col].median()
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        df = df.with_columns(
                            ((pl.col(col) - median) / (Q3 - Q1)).alias(col)
                        )
    return df



def handle_outliers(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    if config["preprocessing_pipeline"]["handle_outliers"]["enabled"]:
        params = config["preprocessing_pipeline"]["handle_outliers"]["params"]
        threshold = params.get("threshold", 3)  # Default Z-score threshold = 3
        
        threshold = params.get("threshold", 3)
        if not isinstance(threshold, (int, float)):
            threshold = 3
            
        methods_dict = params.get("method", {}) 
        
        num_cols = [col for col, dtype in df.schema.items() if dtype in [pl.Float32, pl.Int32, pl.Float64, pl.Int64]]

        for method, cols in methods_dict.items():
            for col in cols:
                if col in num_cols:
                     
                    target_dtype = pl.Float64 if df[col].dtype in [pl.Float32, pl.Float64] else pl.Int64 
                     
                    if method == "z-score":
                        mean = df[col].mean()
                        std = df[col].std()
                        df = df.with_columns(
                            pl.when(((df[col] - mean) / std).abs() > threshold)
                            .then(mean)
                            .otherwise(pl.col(col))
                            .cast(target_dtype)
                            .alias(col)
                        )
        
                    elif method == "iqr":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df.filter((df[col] >= lower_bound) & (df[col] <= upper_bound))

                    elif method == "winsorization":
                        lower_bound = df[col].quantile(0.05)  # Winsorize bottom 5%
                        upper_bound = df[col].quantile(0.95)  # Winsorize top 5%
                        df = df.with_columns(
                            pl.col(col)
                            .map_elements(lambda x: min(max(x, lower_bound), upper_bound), return_dtype=target_dtype)
                            .alias(col)
                        )
    return df


def fill_missing_values(df: pl.DataFrame, config: dict) -> pl.DataFrame:
     if not config["preprocessing_pipeline"]["fill_missing_values"]["enabled"]:
        return df 

     params = config["preprocessing_pipeline"]["fill_missing_values"]["params"]
     strategy = params.get("strategy", None)

     supported_strategies = {"forward", "backward", "min", "max", "mean", "zero", "median", "mode", "constant"}

     if strategy not in supported_strategies:
          raise ValueError(f"Invalid fill strategy: {strategy}. Choose from {supported_strategies}")

     num_cols = [ col for col, dtype in df.schema.items() if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] ]
     cat_cols = [ col for col, dtype in df.schema.items() if dtype == pl.String ]

     if strategy in {"forward", "backward", "min", "max", "zero","one"}: # Although 'mean' strategy is listed in the documentation, we have to specify a case
          df = df.fill_null(strategy=strategy)                             #  for 'mean' to avoid "polars.exceptions.InvalidOperationError: fill-null strategy Mean is not supported"
                                                                           # Note: Probably come back when the issue is resolved
     elif strategy == "mean":
        df = df.with_columns([
            pl.col(col).fill_null(df[col].mean()) for col in num_cols if df[col].null_count() > 0
        ])

     elif strategy == "median":
        median_values = {col: df[col].median() for col in num_cols if df[col].null_count() > 0}
        if median_values:
            df = df.with_columns([pl.col(col).fill_null(median_values[col]) for col in median_values])

     elif strategy == "mode":
          mode_values = {}
          for col in num_cols + cat_cols:
               mode_series = df[col].mode()
               if mode_series.shape[0] > 0:
                    mode_values[col] = mode_series[0]
          if mode_values:
            df = df.with_columns([
                pl.when(pl.col(col).is_null())
                .then(pl.lit(mode_values[col]))
                .otherwise(pl.col(col))
                .alias(col)
                for col in mode_values
            ])

     elif strategy == "constant":
          constant_val = params.get("constant_value", None)
          if constant_val is not None:
               df = df.with_columns([pl.col(col).fill_null(pl.lit(constant_val)) for col in df.columns])

     return df



def custom_enforce_types(df : pl.DataFrame, config : dict) -> pl.DataFrame:

    if not config['preprocessing_pipeline']['custom_enforce_types']['enabled']:
        return df
    
    # Get function config from yaml
    feature_groups = config['preprocessing_pipeline']['custom_enforce_types']['params']['per_column']

    # Process columns
    if len(feature_groups['int']) != 0:
        df = df.with_columns(pl.col(c).cast(pl.Int64) for c in feature_groups['int'])
    if len(feature_groups['float']) != 0:
        df = df.with_columns(pl.col(c).cast(pl.Float64) for c in feature_groups['float'])
    if len(feature_groups['string']) != 0:
        df = df.with_columns(pl.col(c).cast(pl.String) for c in feature_groups['string'])
    if len(feature_groups['datetime']) != 0:
        df = df.with_columns(pl.col(c).str.to_datetime() for c in feature_groups['datetime'])
    return df



def lowercase_all_values(df : pl.DataFrame, config : dict) -> pl.DataFrame:
    if not config['preprocessing_pipeline']['lowercase_all_values']['enabled']:
        return df
    return df.with_columns(pl.col(pl.String).str.to_lowercase())



def _custom_literal_eval(x : str, include_cols : list[str]):

    # Get dictionary in string
    d = ast.literal_eval(x)
    
    # Normalize columns to include_cols
    d = {key : val for key, val in d.items() if key in include_cols}
    d = {key : d.get(key, None) for key in include_cols}

    # Check if val_type is list. If so, join it with commas
    for key, val in d.items():
        if isinstance(val, list):
            d[key] = ','.join(val)
    return d



def expand_dict_columns(df, config):
    
    if not config['preprocessing_pipeline']['expand_dict_columns']['enabled']:
        return df
    
    # Get config params
    params = config['preprocessing_pipeline']['expand_dict_columns']['params']
    dict_cols = params['dict_cols']
    include_cols_dict = params['include_cols']

    for col in dict_cols:

        # Make schema
        struct_schema = pl.Struct([pl.Field(c, pl.String) for c in include_cols_dict[col]])

        # Expand dict columns
        df = df.with_columns(pl.col(col)
            .map_elements(lambda x : _custom_literal_eval(x, include_cols_dict[col]), return_dtype = struct_schema)
            .alias('struct')).unnest('struct')
    return df

