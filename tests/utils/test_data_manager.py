import pandas as pd
import pandas.api.types as ptypes
import numpy as np

from concrete_strength_prediction.config.core import config

class TestDataManager(object):
        
    def test_columns_names_are_based_on_mapping(self, dm):
        assert np.array_equal(dm.df.columns.tolist(), list(config.config_model.cols_mapping.values()))
        
    def test_imports_as_dataframe(self, dm):
        assert isinstance(dm.df, pd.DataFrame)
        assert isinstance(dm.get_data(), pd.DataFrame)
 
    def test_data(self, dm):
        assert len(dm.df.shape) == 2
        assert dm.df.shape[1] == len(config.config_model.cols_mapping.keys())
        assert all([ptypes.is_numeric_dtype(dm.df[col])  for col in dm.df.columns])
