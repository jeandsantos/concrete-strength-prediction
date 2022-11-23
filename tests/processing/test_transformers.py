import numpy as np
import pytest

from concrete_strength_prediction.processing.transformers import (
    CustomTransformerError, PercentageTransformer, RatioTransformer,
    SumTransformer)


class TestSumTransformer(object):
    def test_transform_with_standard_input(self, df_standard):

        st = SumTransformer(columns=["a", "b", "c"], col_name="sum_test")
        st_out = st.transform(df_standard)

        assert pytest.approx(st_out["sum_test"]) == np.array([6.0, 10.0, 16.0])
        assert "sum_test" in st_out.columns

    def test_transform_without_specifying_columns(self):
        with pytest.raises(TypeError):
            st = SumTransformer(col_name="sum_test")

    def test_transform_without_specifying_columns(self):
        with pytest.raises(CustomTransformerError):
            st = SumTransformer(columns=None)

    def test_transform_with_wrong_data_types(self):
        with pytest.raises(CustomTransformerError):
            st = SumTransformer(columns="None")

    def test_transform_with_empty_list(self):
        with pytest.raises(CustomTransformerError):
            st = SumTransformer(columns=[])
