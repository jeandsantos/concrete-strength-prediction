from concrete_strength_prediction.config.core import (
    PATH_CONFIG_FILE, 
    get_config_file_path, 
    get_config_from_file, 
    get_and_validate_config
)

from pathlib import Path
from strictyaml import YAML
import pytest
from pydantic import BaseModel

class TestGetConfigFilePath(object):
    
    def test_valid_path(self):
        
        config_file_path = get_config_file_path()
        
        assert config_file_path == PATH_CONFIG_FILE
        assert isinstance(config_file_path, Path)
        assert config_file_path.is_file()
        assert config_file_path.exists()
        
    # TODO adds tests for invalid/unexisting config file
    
    
class TestGetConfigFromFile(object):
    
    def test_without_adding_config_path(self):
        
        config = get_config_from_file()
        
        assert isinstance(config, YAML)
        assert isinstance(config.data, dict)
        assert isinstance(config.data['path_data'], str)
        assert isinstance(config.data['bool_verbose'], str)
    
    def test_with_adding_config_path(self):
        
        config = get_config_from_file(PATH_CONFIG_FILE)
        
        assert isinstance(config, YAML)
        assert isinstance(config.data, dict)
        assert isinstance(config.data['path_data'], str)
        assert isinstance(config.data['bool_verbose'], str)
    
    def test_with_unexisting_config_path(self):
        
        with pytest.raises(FileNotFoundError):
            get_config_from_file('./does_not_exist.yml')

class TestGetAndValidateConfig(object):
    
    def test_output_type(self):
        
        config = get_and_validate_config()
        
        assert isinstance(config, BaseModel)
        assert isinstance(config.config_app.path_data, str) 
        assert config.config_app.bool_verbose in [True, False] 
    
    # TODO adds tests for invalid YAML