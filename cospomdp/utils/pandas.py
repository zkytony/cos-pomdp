# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
