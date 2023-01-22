import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from round_result import round_result

def standard_scaled(dataFrame, add_const_column = False):
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(dataFrame.values), index=dataFrame.index, columns=dataFrame.columns)
    if add_const_column: scaled['const'] = 1
    return scaled

def cross_validate(model, X_train_chosen, y_train, iterations = 5, random_seed = 5):

    train_errors = []
    validate_errors = []

    for i in range(iterations):
        X_train_cross, X_val_cross, y_train_cross, y_val_cross = train_test_split(X_train_chosen, y_train, test_size=0.2, random_state=random_seed*i)

        model.fit(X_train_cross, y_train_cross)

        train_errors.append(metrics.mean_squared_error(y_train_cross, model.predict(X_train_cross)))
        validate_errors.append(metrics.mean_squared_error(y_val_cross, model.predict(X_val_cross)))

    avg_train_error = np.mean(train_errors)
    avg_validate_error = np.mean(validate_errors)

    return (avg_train_error, avg_validate_error)


def describe_multiple_regression_model(X_train, X_test, y_train, y_test, considered_features):
    X_train_chosen = X_train[considered_features]
    model = multiple_regression_model(X_train_chosen, y_train)
    (avg_train_error, avg_val_error) = model.cross_validate(X_train_chosen, y_train)
    print(f"\nAvg train error =\t{avg_train_error}")
    print(f"Avg validate error =\t{avg_val_error}")
    print(f"Test error =\t\t{round_result(y_test, model.predict(X_test[considered_features]))[1]['actual_mean_sqrt']}\n")


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

            model = multiple_regression_model(X_train_cross, y_train_cross)
            
            train_errors.append(metrics.mean_squared_error(y_train_cross, model.predict(X_train_cross)))
            validate_errors.append(metrics.mean_squared_error(y_val_cross, model.predict(X_val_cross)))

        avg_train_error = np.mean(train_errors)
        avg_validate_error = np.mean(validate_errors)

        return (avg_train_error, avg_validate_error)
