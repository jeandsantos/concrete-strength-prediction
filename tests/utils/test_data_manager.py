import pandas as pd
import numpy as np
import pytest
from concrete_strength_prediction.utils.data_manager import DataManager 

path_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'

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

# TODO create fixture
dm = DataManager(path_data, cols_mapping)

def test_assert_true():
    assert True    
class TestDataManager(object):

    
    def test_columns_names_are_based_on_mapping(self):
        np.array_equal(dm.df.columns.tolist(), list(cols_mapping.values()))
    dm.load_dataset()
        
    def test_imports_as_dataframe(self):
        assert isinstance(dm.df, pd.DataFrame)
        assert isinstance(dm.get_data(), pd.DataFrame)