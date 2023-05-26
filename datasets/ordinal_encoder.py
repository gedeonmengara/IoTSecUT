from __future__ import absolute_import, division, print_function, unicode_literals

# Modified https://github.com/tcassou/mlencoders/blob/master/mlencoders/base_encoder.py

import numpy as np
import pandas as pd
import pickle

NAN_CATEGORY = 0

class OrdinalEncoder():
    """
    Target Encoder for categorical features.
    """

    def __init__(self, cols=None, handle_unseen='ignore', imputed=0, min_sample=1):

        """
        cols: [list of strings] : list of columns to encode or None (all field will be encoded)
        handle_unseen: [string] : 
                                    "error" : raise an error if a category is unseen
                                    "ignore" : skip unseen categories
                                    "impute" : impute new categories to a predefined value
        """

        self._input_check("handle_unseen", handle_unseen, ["error", "ignore", "impute"])

        self.cols = cols
        self.handle_unseen = handle_unseen
        self.min_samples = min_sample
        self.imputed = imputed
        self._mapping = {}

    def _input_check(self, name, value, options):
        if value not in options:
            raise ValueError("Wrong input: {} parameter must be in {}".format(name, options))

    def check_columns(self, X, y):
        if self.cols is None:
            self.cols = X.columns
        else:
            assert all(c in X.columns for c in self.cols)

        if y is not None:
            assert X.shape[0] == y.shape[0]

    def fit(self, X, y=None):
        """
        X : [DataFrame] : must contains columns
        """

        self.check_columns(X, y)

        for col in self.cols:
            map = (
                pd.Series(pd.unique(X[col].fillna(NAN_CATEGORY)), name=col)
                .reset_index()
                .rename(columns={"index": "value"})
            )

            map["value"] += 1
            self._mapping[col] = map.set_index(col)

    def transform(self, X):

        X_encoded = X.copy(deep=True)

        for col, mapping in self._mapping.items():
            X_encoded.loc[:, col] = (
                X_encoded[col].fillna(NAN_CATEGORY).map(mapping["value"])
            )

            if self.handle_unseen == "impute":
                X_encoded[col].fillna(self.imputed, inplace=True)

            elif self.handle_unseen == "error":
                if np.unique(X_encoded[col]).shape[0] > mapping.shape[0]:
                    raise ValueError("Unseen categories found in `{}` column.".format(col))

        return X_encoded

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def save_mapping(self, path):
        pickle.dump(self.__dict__, open(path, "wb"))

    def load_mapping(self, path):
        for k, v in pickle.load(open(path, "rb")).items():
            setattr(self, k, v)