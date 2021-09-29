from sklearn.base import BaseEstimator, TransformerMixin
import logging
import numpy as np
import pandas as pd

class DotProduct(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        self.X_transformed = self.fit_transform(X)

        return self 

    def fit_transform(self, X, y=None):
        self.X_index = X.index

        # dot product to get similar movies
        X_sub = np.array(X.iloc[:,4:])
        dot_prod = X_sub.dot(np.transpose(X_sub))

        logging.debug(f"Dot product done. Shape {dot_prod.shape}.")

        X_transformed = pd.DataFrame(dot_prod)
        X_transformed.columns = X.index.tolist()
        X_transformed[X.index.name] = self.X_index
        X_transformed.set_index(X.index.name, inplace=True)

        logging.debug("Dataframe restructuring done.")

        return X_transformed