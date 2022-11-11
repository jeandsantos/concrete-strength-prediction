import pytest

from concrete_strength_prediction.utils.data_manager import DataManager
from concrete_strength_prediction.config.core import config

@pytest.fixture(scope='module')
def dm():
    
    dm = DataManager(
        path=config.config_app.path_data, 
        columns_mapping=config.config_model.cols_mapping
        )
    dm.load_dataset()

    yield dm