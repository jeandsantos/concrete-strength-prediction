import numpy as np
import pandas as pd
import pytest

from concrete_strength_prediction.processing.transformers import (
    CustomTransformerError, PercentageTransformer, RatioTransformer,
    SumTransformer)


class TestSumTransformer(object):
    def test_transform_with_standard_input(self, df_standard):

        tr = SumTransformer(columns=["a", "b", "c"], col_name="sum_test")
        tr_out = tr.transform(df_standard)

        assert pytest.approx(tr_out["sum_test"]) == np.array([6.0, 10.0, 16.0])
        assert "sum_test" in tr_out.columns

    def test_transform_without_specifying_columns(self):
        with pytest.raises(TypeError):
            tr = SumTransformer(col_name="sum_test")

    def test_transform_without_specifying_columns(self):
        with pytest.raises(CustomTransformerError):
            tr = SumTransformer(columns=None)

    def test_transform_with_wrong_data_types(self):
        with pytest.raises(CustomTransformerError):
            tr = SumTransformer(columns="None")

    def test_transform_with_empty_list(self):
        with pytest.raises(CustomTransformerError):
            tr = SumTransformer(columns=[])


class TestPercentageTransformer(object):
    def test_transform_with_all_columns(self, df_standard):

        tr = PercentageTransformer(["a", "b", "c"])
        tr_out = tr.transform(df_standard)
        
        tr_out_expected = pd.DataFrame(
            [
                [16.666666, 33.333333, 50.],
                [60., 30., 10.],
                [62.5,  6.25, 31.25]
            ]
        )
        
        assert pytest.approx(tr_out) == tr_out_expected

    def test_transform_with_some_columns(self, df_standard):

        tr = PercentageTransformer(["a", "c"])
        tr_out = tr.transform(df_standard)
        
        tr_out_expected = pd.DataFrame(
            [
                [25.        ,  2.        , 75.        ],
                [85.71428571,  3.        , 14.28571429],
                [66.66666667,  1.        , 33.33333333]
            ]
        )
        
        assert pytest.approx(tr_out) == tr_out_expected
        
    def test_transform_without_specifying_numerator_columns(self):
        with pytest.raises(CustomTransformerError):
            tr = PercentageTransformer(col_numerator=None)

    def test_transform_with_wrong_data_types(self):
        with pytest.raises(CustomTransformerError):
            tr = PercentageTransformer(col_numerator="None")

    def test_transform_with_empty_list(self):
        with pytest.raises(CustomTransformerError):
            tr = PercentageTransformer(col_numerator=[])


class TestRatioTransformer(object):
        
    @pytest.mark.parametrize('col_numerator, col_denominator, result',
                             [
                                 (['a', 'b'], ['c',], pd.Series([1. , 9. , 2.2])),
                                 (['a',], ['b',], pd.Series([0.5,  2. , 10.])),
                             ])
    def test_transform_with_parameters(self, col_numerator, col_denominator, result, df_standard):

        tr = RatioTransformer(col_numerator, col_denominator)
        tr_out = tr.transform(df_standard)
                
        assert pytest.approx(tr_out['cols_ratio']) == result
        
    def test_transform_without_specifying_numerator_columns(self):
        with pytest.raises(CustomTransformerError):
            tr = RatioTransformer(col_numerator=None, col_denominator= ['b',])

    def test_transform_with_wrong_data_types_numerator(self):
        with pytest.raises(CustomTransformerError):
            tr = RatioTransformer(col_numerator="None", col_denominator= ['b',])

    def test_transform_with_wrong_data_types_denominator(self):
        with pytest.raises(CustomTransformerError):
            tr = RatioTransformer(col_numerator=['a',], col_denominator='None')

    def test_transform_with_empty_list_numerator(self):
        with pytest.raises(CustomTransformerError):
            tr = RatioTransformer(col_numerator=[], col_denominator= ['b',])

    def test_transform_with_empty_list_denominator(self):
        with pytest.raises(CustomTransformerError):
            tr = RatioTransformer(col_numerator=['a',], col_denominator=[])
