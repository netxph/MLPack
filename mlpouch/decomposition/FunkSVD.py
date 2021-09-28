from sklearn.base import BaseEstimator, TransformerMixin
import logging

class FunkSVD(TransformerMixin, BaseEstimator):
    
    def __init__(self, latent_features = 4, learning_rate = 0.0001, iters = 100):
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

    def fit(self, X, y = None, sample_weight=None):
        return self

    def score(self, X, y=None):
        pass