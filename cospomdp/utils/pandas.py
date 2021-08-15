# Utils for pandas
import pandas as pd

def flatten_index(df_or_s):
    """Flatten multi-index into columns and return a new DataFrame.
    Note: Even if the input is a Series with multi-index, the output
    will be a DataFrame.

    Arg:
        df: DataFrame or Series
    """
    if isinstance(df_or_s, pd.DataFrame):
        df = df_or_s
        df.columns = list(map("-".join, df.columns.values))
        return df.reset_index()
    else:
        s = df_or_s
        return s.reset_index()
