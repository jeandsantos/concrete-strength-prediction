# A feature engineering pipeline for the transformation of both numeric and non-numeric data

from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from processing.transformers import (
    PercentageTransformer,
    RatioTransformer,
    SumTransformer,
)

from config.core import config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocessor = Pipeline(
    steps=[
        # Create percentages
        (
            "percentage",
            PercentageTransformer(col_numerator=config.config_model.cols_composition),
        ),
        ## Feature Creation
        # Create sum of solids
        (
            "total_solids",
            SumTransformer(
                columns=config.config_model.cols_solids, col_name="total_solids"
            ),
        ),
        # Create ratio-based features
        (
            "ratio_aggregates_solids",
            RatioTransformer(
                col_numerator=config.config_model.cols_ratio_aggregates_solids_num,
                col_denominator=config.config_model.cols_ratio_aggregates_solids_den,
                name="ratio_aggregates_solids",
            ),
        ),
        # Create ratio (married to not married)
        (
            "ratio_cement_water",
            RatioTransformer(
                col_numerator=config.config_model.cols_ratio_cement_water_num,
                col_denominator=config.config_model.cols_ratio_cement_water_den,
                name="ratio_cement_water",
            ),
        ),
        ## Feature Transformation
        # Yeo-Johnson Transform
        (
            "yeojohnson", 
            YeoJohnsonTransformer()
        ),
        # Z-score scaling
        (
            "standardization", 
            SklearnTransformerWrapper(transformer=StandardScaler())
        ),
        # Remove highly correlated
        (
            "remove_correlated",
            SmartCorrelatedSelection(
                **config.config_model.fe_smart_correlation_params.dict()
            ),
        ),
        ## Feature Selection
        # Drop Constant Features
        (
            "drop_constant",
            DropConstantFeatures(**config.config_model.fe_drop_constant_params.dict()),
        ),
    ]
)
