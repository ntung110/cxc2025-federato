
import os
import re
import logging
from pathlib import Path
import polars as pl
from collections.abc import Callable


def setup_logging(log_path):
    '''
    Set up logging to track pipeline execution.
    '''
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(filename=log_path,
                        level=logging.INFO,
                        format= "%(asctime)s - %(levelname)s - %(message)s",
                        )


def load_data(config):
    return pl.read_parquet(config["pipeline"]["data_source"])


def _file_sort(dir : list[str]):
    files = [f for f in dir if 'df' in f]
    sorted_files = sorted(files, key = lambda x : int(re.findall(r'\d+', x)[0]))
    return sorted_files


def batch_preprocess_data(df : pl.DataFrame, func : Callable, 
                          config : dict, chunk_size : int) -> pl.DataFrame:
    if not config['preprocessing_pipeline']['batch_preprocess_data']['enabled']:
        return df
    
    # Get path for data input and output
    input_path = config['pipeline']['input_path']
    output_path = config['pipeline']['output_path']

    # Sequentially apply function to chunks of data and export them 
    for idx in range(0, df.shape[0], chunk_size):
        df_chunk = df[idx : min(df.shape[0], idx + chunk_size)]
        df_chunk = func(df_chunk, config)
        df_chunk.write_parquet(f'df_{idx}.parquet')
    del df_chunk

    # Load chunk again and start concatenating
    df_combined = None
    for file in _file_sort(os.listdir(config['pipeline']['output_path'])):
        if 'df' not in file:
            continue
        df_chunk = pl.read_parquet(output_path + file)
        
        if df_combined is None:
            df_combined = df_chunk
            expected_cols = df_chunk.columns
        else: 

            # There might be columns missing from chunk
            # If so, add columns in and fill all of the values with null
            missing_cols = set(expected_cols) - set(df_chunk.columns)
            extra_cols = set(df_chunk.columns) - set(expected_cols)
            df_chunk = df_chunk.with_columns(pl.lit(None).alias(c) for c in missing_cols)

            df_chunk = df_chunk.select(expected_cols)
            df_combined = pl.concat([df_combined, df_chunk], how = 'vertical_relaxed')
    
    del df_chunk
    return df_combined


def export_data(df : pl.DataFrame, config : dict) -> None:
    output_path = config['pipeline']['output_path']
    logging.info(f"Saved processed data to {config['pipeline']['output_path']}")