import polars as pl

class Dataloader:

    """ Dataloader is an iterator which loads data sequentially in batches"""

    def __init__(self, batch_size : int) -> None:
        self.batch_size = batch_size
        self._index = 0

    def fit(self, X : pl.DataFrame, y : pl.Series) -> None:
        self.X = X
        self.y = y
        self.len = len(self.y)

    def __iter__(self):
        return self
   
    def __next__(self):
       
        if self._index >= self.len:
            raise StopIteration

        start = self._index
        end = min(self.len, self._index + self.batch_size)

        X_batch = self.X[start : end]
        y_batch = self.y[start : end]
        self._index += self.batch_size

        return X_batch, y_batch
    

class TrainValidationSplit:

    """
    TrainValidationSplit is a class which handles the splitting of data
    into training and validation sets using a specified ratio
    """

    def __init__(self, val_ratio : float, random_state : int = None):
        self.val_ratio = val_ratio
        self.train_ratio = 1 - self.val_ratio
        self.random_state = random_state

        # Assert value range
        assert self.train_ratio > 0, 'specfied ratio out of range'
        assert self.train_ratio < 1, 'specified ratio out of range'

    def X_y_split(self, df, target_col):
        return df.drop(target_col), df[target_col]

    def data_split(self, df):

        # Split train and validation sets
        df = df.sample(fraction = 1, shuffle = True)
        val_num_rows = int(self.val_ratio * df.shape[0])      
        df_val, df_train = df.head(val_num_rows), df.tail(-val_num_rows)

        return df_train, df_val


class TimeSplit:

    """
    TimeSplit is a class which handles the splitting of data into a test set
    """

    def __init__(self, time_cutoff : str) -> None:
        self.time_cutoff = time_cutoff


    def X_y_split(self, df, target_col):
        return df.drop(target_col), df[target_col]

    def data_split(self, df, time_col):
        # Split test data by time cutoff
        # df = df.with_columns(
        #     pl.col(time_col).str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f")
        # )        
        parsed_time_cutoff = pl.lit(self.time_cutoff).str.to_datetime(format="%Y-%m-%d") 
        df_test = df.filter(pl.col(time_col) > parsed_time_cutoff)
        df_leftover = df.filter(pl.col(time_col) <= parsed_time_cutoff)
        return df_test, df_leftover
