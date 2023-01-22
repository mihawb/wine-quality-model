import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split


class multiple_regression_model:

    def __init__(self, X_chosen_features, y):
        pinw = np.linalg.pinv(X_chosen_features)
        self.model = np.dot(pinw, y)

    def predict(self, X_chosen_features):
        return np.sum(X_chosen_features * self.model, axis=1)

    def cross_validate(self, X_train_chosen, y_train, iterations = 5, random_seed = 5):

        train_errors = []
        validate_errors = []
        for i in range(iterations):
            X_train_cross, X_val_cross, y_train_cross, y_val_cross = train_test_split(X_train_chosen, y_train, test_size=0.2, random_state=random_seed*i)

            model = multiple_regression_model(X_train_chosen, y_train)
            
            train_errors.append(metrics.mean_squared_error(y_train_cross, self.predict(X_train_cross)))
            validate_errors.append(metrics.mean_squared_error(y_val_cross, self.predict(X_val_cross)))

        avg_train_error = np.mean(train_errors)
        avg_validate_error = np.mean(validate_errors)

        return (avg_train_error, avg_validate_error)
