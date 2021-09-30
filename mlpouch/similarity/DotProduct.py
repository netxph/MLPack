from sklearn.base import BaseEstimator, TransformerMixin
import logging
import numpy as np
import pandas as pd

class DotProduct(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        dot_prod = self.fit_transform(X)

        X_transformed = pd.DataFrame(dot_prod)
        X_transformed.columns = X.index.tolist()
        X_transformed[X.index.name] = self.X_index
        X_transformed.set_index(X.index.name, inplace=True)

        self.X_transformed = X_transformed

        return self 

    def fit_transform(self, X, y=None):
        self.X_index = X.index

        # dot product to get similar movies
        X_arr = np.array(X)
        dot_prod = X_arr.dot(np.transpose(X_arr))

        return dot_prod