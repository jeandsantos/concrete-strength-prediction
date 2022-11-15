from pathlib import Path

import pandas as pd


class DataManager:
    """A Class to import and transform data"""

    def __init__(self, path: str, columns_mapping: dict) -> None:

        self._path = path
        self._columns_mapping = columns_mapping

    def load_dataset(self, *args, **kwargs) -> None:
        """Load data and save it as an attribute

        Returns:
            None: None
        """

        if Path(self._path).suffix in [".xls", ".xlsx"]:
            print("Importing files as Excel file")
            self._df = pd.read_excel(self._path, *args, **kwargs)
        else:
            print("Importing files as CSV file")
            self._df = pd.read_csv(self._path, *args, **kwargs)

        if self._columns_mapping is not None:
            self._df = self._df.rename(columns=self._columns_mapping)

    def get_data(self) -> pd.DataFrame:

        return self._df

    @property
    def df(self):
        return self._df

    @property
    def path(self):
        return self._path

    @property
    def columns_mapping(self):
        return self._columns_mapping
