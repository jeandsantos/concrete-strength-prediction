import pandas as pd
import pandas.api.types as ptypes
import numpy as np

cols_mapping = {
    'Cement (component 1)(kg in a m^3 mixture)': 'cement',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'slag',
    'Fly Ash (component 3)(kg in a m^3 mixture)': 'ash',
    'Water  (component 4)(kg in a m^3 mixture)': 'water',
    'Superplasticizer (component 5)(kg in a m^3 mixture)': 'superplasticizer',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'coarse_aggregate',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'fine_aggregate',
    'Age (day)': 'age',
    'Concrete compressive strength(MPa, megapascals) ': 'strength',
}

class TestDataManager(object):
        
    def test_columns_names_are_based_on_mapping(self, dm):
        assert np.array_equal(dm.df.columns.tolist(), list(cols_mapping.values()))
        
    def test_imports_as_dataframe(self, dm):
        assert isinstance(dm.df, pd.DataFrame)
        assert isinstance(dm.get_data(), pd.DataFrame)
 
    def test_data(self, dm):
        assert len(dm.df.shape) == 2
        assert dm.df.shape[1] == len(cols_mapping.keys())
        assert all([ptypes.is_numeric_dtype(dm.df[col])  for col in dm.df.columns])
