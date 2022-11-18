import pytest
import pandas as pd

from concrete_strength_prediction.config.core import config
from concrete_strength_prediction.utils.data_manager import DataManager


@pytest.fixture(scope="module")
def dm():

    dm = DataManager(
        path=config.config_app.path_data,
        columns_mapping=config.config_model.cols_mapping,
    )
    dm.load_dataset()

    yield dm

@pytest.fixture
def df_standard():

    df_standard = pd.DataFrame(
        [[1,2,3], [6,3,1], [10,1,5],], 
        columns=['a', 'b', 'c',], 
        dtype=float)

    yield df_standard