from pathlib import Path
from typing import Dict, List, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

# Directories
PATH_PACKAGE = Path(__file__).resolve().parent.parent
PATH_CONFIG_FILE = PATH_PACKAGE / "config.yml"
PATH_MODELS = PATH_PACKAGE / "trained_models"

# Feature Engineering Parameters
class SmartCorrelationModel(BaseModel):
    threshold: float
    method: str
    selection_method: str
    missing_values: str


class ConstantParamsModel(BaseModel):
    tol: int
    missing_values: str


# Configurations
class AppConfig(BaseModel):
    """Application-level configuration"""

    project_name: str
    path_data: str
    mlflow_bool: bool
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    bool_verbose: bool


class ModelConfig(BaseModel):
    """All configurations related to the model"""

    cols_mapping: Dict[str, str]
    cols_features: List[str]
    cols_composition: List[str]
    cols_solids: List[str]
    cols_ratio_aggregates_solids_num: List[str]
    cols_ratio_aggregates_solids_den: List[str]
    cols_ratio_cement_water_num: List[str]
    cols_ratio_cement_water_den: List[str]
    cols_target: str
    fe_drop_constant_params: ConstantParamsModel
    fe_smart_correlation_params: SmartCorrelationModel
    test_size: float
    stratify_by: str
    search_iterations: int
    cv_scores: Sequence
    cv_metric: str
    cv_folds: int
    seed: int
    n_jobs: int
    n_features: int


class Config(BaseModel):
    """Master Configuration Object"""

    config_app: AppConfig
    config_model: ModelConfig


def get_config_file_path() -> Path:
    """
    Locate and return the path of the configuration file
    """

    if PATH_CONFIG_FILE.is_file():
        return PATH_CONFIG_FILE

    raise FileNotFoundError(f"Configuration file not found at {PATH_CONFIG_FILE!r}")


def get_config_from_file(path_config: Path = None) -> YAML:
    """
    Parse YAML file containing the package configuration
    """

    if not path_config:
        path_config = get_config_file_path()

    if path_config:
        with open(path_config, "r") as file:
            config_parsed = load(file.read())

            return config_parsed

    raise FileNotFoundError(f"Did not find config file at path: {path_config}")


def get_and_validate_config(config_parsed: YAML = None) -> Config:
    """
    Run validation on config values
    """

    if config_parsed is None:
        config_parsed = get_config_from_file()

    config_validated = Config(
        config_app=AppConfig(**config_parsed.data),
        config_model=ModelConfig(**config_parsed.data),
    )

    return config_validated


config = get_and_validate_config()
print("Validation Completed")
