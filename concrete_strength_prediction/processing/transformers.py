from typing import List

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformerError(Exception):
    """An exception class for SumTransformer"""


class SumTransformer(BaseEstimator, TransformerMixin):
    """
    Create feature with the sum of the values of selected columns
    """

    def __init__(self, columns: List[str], col_name: str = "cols_sum") -> None:

        if not isinstance(columns, list):
            raise CustomTransformerError(f"'columns' must be a list")

        if len(columns) == 0:
            raise CustomTransformerError(f"'columns' must be at least of length 1")

        self.columns = columns
        self.col_name = str(col_name)

    def fit(self, X: DataFrame, y=None):
        return self

    def transform(self, X: DataFrame):
        X = X.copy()

        values = X[self.columns].sum(axis=1)

        X[self.col_name] = values

        return X


class PercentageTransformer(BaseEstimator, TransformerMixin):
    """
    Converts counts to percentages
    """

    def __init__(
        self, col_numerator: List[str], col_denominator: List[str] = None
    ) -> None:

        if not isinstance(col_numerator, list):
            raise CustomTransformerError("col_numerator must be a list")

        self.col_numerator = col_numerator
        self.col_denominator = col_denominator

    def fit(self, X: DataFrame, y=None):
        return self

    def transform(self, X: DataFrame):
        X = X.copy()

        # Use the sum of all columns as the denominator if not columns are provided to col_denominator
        if self.col_denominator is None:
            denomitator = X[self.col_numerator].sum(axis=1)

        elif isinstance(self.col_denominator, list):
            denomitator = X[self.col_denominator].sum(axis=1)

        X[self.col_numerator] = X[self.col_numerator].div(denomitator, axis=0).mul(100)

        return X


class RatioTransformer(BaseEstimator, TransformerMixin):
    """
    Calculates ratios between one or more columns of a DataFrame
    """

    def __init__(
        self,
        col_numerator: List[str],
        col_denominator: List[str],
        name: str = "cols_ratio",
    ) -> None:

        if not isinstance(col_numerator, list):
            raise CustomTransformerError("col_numerator must be a list")

        if not isinstance(col_denominator, list):
            raise CustomTransformerError("col_denominator must be a list")

        self.col_numerator = col_numerator
        self.col_denominator = col_denominator
        self.name = name

    def fit(self, X: DataFrame, y=None):
        return self

    def transform(self, X: DataFrame):

        X = X.copy()

        numerator = X[self.col_numerator].sum(axis=1)
        denomitator = X[self.col_denominator].sum(axis=1)

        X[self.name] = np.divide(numerator, denomitator)

        return X
