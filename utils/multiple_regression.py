import numpy as np


class multiple_regression_model:

    def __init__(self, X_chosen_features, y):
        pinw = np.linalg.pinv(X_chosen_features)
        self.model = np.dot(pinw, y)

    def predict(self, X_chosen_features):
        return np.sum(X_chosen_features * self.model, axis=1)
